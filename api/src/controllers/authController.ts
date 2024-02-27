import {Request, Response} from 'express';
import {v4 as uuidv4} from 'uuid';
import {getConnection, getCustomRepository, In} from 'typeorm';
import {
    Token,
    generateSignToken,
    generateRefreshToken,
    verifyToken,
    verifyRefreshToken,
} from '../domain/Session';
import crypto from 'crypto';
import moment from 'moment';
import {unitOfTime} from 'moment';
import config from '../config/config';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {User} from '../infrastructure/entity/Users';
import {RefreshToken} from '../infrastructure/entity/RefreshToken';
import {UserRoleTypes, isEitherRolesMatch} from '../domain/UserRoles';
import {Roles} from '../infrastructure/entity/Roles';
import argon2 from 'argon2';
import {UserRepository} from '../infrastructure/repositories/userRepo';
import {RolesRepository} from '../infrastructure/repositories/rolesRepo';
import {UserLogin} from '../domain/vo/UserLogin';
import {RefreshTokenRepository} from '../infrastructure/repositories/refreshTokenRepo';
import {notificationSmsService, notificationEmailService, localeService, pinpoint, fileService} from '../container';
import {VerificationCodes} from '../infrastructure/entity/VerificationCodes';
import {VerificationCodesRepository} from '../infrastructure/repositories/verificationCodesRepo';
import {DoctorsPatients} from '../infrastructure/entity/DoctorsPatients';
import {getUserRoles} from '../middlewares/checkRole';
import {PersonalData} from '../infrastructure/entity/PersonalData';
import {Gender} from '../domain/Gender';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';

type RegisterUserBody = {
    login: string,
    email: string,
    password: string,
    phoneNumber?: string,
    comment?: string,
    roles?: string[],
    patients?: string[]
}

type PatchUserBody = {
    password?: string,
    phoneNumber?: string,
    comment?: string,
    roles?: string[],
    patients?: string[],
    isActive?: boolean,
    isCheckHealthy?: boolean,
    isCheckCovid?: boolean,
    isValidateCough?: boolean,
    checkStart?: string,
    checkEnd?: string
}

type PatchDataBody = {
    identifier?: string,
    age?: number,
    gender?: Gender,
    is_smoking?: boolean
}

export class AuthController {

    private regexpEmail = new RegExp('^.+@.+$');
    private regexpNumber = new RegExp('^[+][0-9]{1,15}$');

    private async validatePhoneNumber (phoneNumber: string) {
        const inputNumber = {
            NumberValidateRequest: { 
                PhoneNumber: phoneNumber
            }
        };
        const pinpointResponse = await pinpoint.phoneNumberValidate(inputNumber).promise();
        if (!['INVALID', 'LANDLINE'].includes(pinpointResponse.NumberValidateResponse.PhoneType)) {
            return Promise.resolve(true);
        } else {
            console.error(`PhoneType: ${pinpointResponse.NumberValidateResponse.PhoneType}`);
            return Promise.resolve(false);
        }
    }

    private async verifyPassword (hash: string, password: string) {
        try {
            const isMatch = await argon2.verify(hash, password);
            return Promise.resolve(isMatch);
        } catch(error) {
            console.error(error);
            return Promise.resolve(false);
        }
    }

    private async generateAndSendCode (user: User, lang: string) {
        const verificationCode = crypto.randomInt(1000, 10000);
        const text = localeService.translate({phrase: 'verification_code', locale: lang});

        const verification = new VerificationCodes();
        verification.user = user;
        verification.code = verificationCode;
        const period = config.verificationCodeLife.match(/[^\d.-]/g).join() as unitOfTime.Base;
        verification.date_expired = moment().add(
            parseInt(config.verificationCodeLife),
            period
        ).toDate();
        
        try {
            const verificationRepo = getCustomRepository(VerificationCodesRepository);
            await verificationRepo.save(verification);
            if (user.phone_number != undefined) {
                const message = `${text}: ${verificationCode}`;
                await notificationSmsService.sendUserSms(user.phone_number, message);
            }
            if (user.email != undefined) {
                if (user.is_email_error) {
                    console.error(`Can not send message to this email due to ${user.email_error_type}`);
                    throw new Error(`Can not send message to this email due to ${user.email_error_type}`);
                }
                const message = `${text}: <p style="font-size:40px"><b>${verificationCode}</b></p>`;
                await notificationEmailService.sendUserEmail(user.email, message, text);
            }
        } catch (error) {
            throw error;
        }
    }

    public async login (req: Request, res: Response) {
        const {email, login, password} = req.body;
    
        if (!((login || email) && password)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const usersRepository = getCustomRepository(UserRepository);
        let user: User;    
        
        try {
        // TODO: check both email and login if present
            user = await usersRepository.findActiveByLoginOrEmailOrFail(login, email);
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!await this.verifyPassword(user.password, password)) {
            const errorMessage = getErrorMessage(HttpErrors.PASSWORD_MISMATCH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        
        const jwtId = uuidv4();
        const userRoles = user.roles.map((roleEntity) => roleEntity.role);
        const token = generateSignToken({userId: user.id, roles: userRoles, jwtId} as Token);
        const refreshToken = await generateRefreshToken(user, jwtId);    
        return res.send({token, refreshTokenId: refreshToken.id});
    }
    
    public async adminLogin (req: Request, res: Response) {
        const {email, login, password} = req.body;
    
        if (!((login || email) && password)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
      
        const usersRepository = getCustomRepository(UserRepository);
        let user: User;    
        
        try {
        // TODO: check both email and login if present
            user = await usersRepository.findActiveByLoginOrEmailOrFail(login, email);
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        const userRoles = user.roles.map((roleEntity) => roleEntity.role);
        if (!isEitherRolesMatch([UserRoleTypes.DATA_SCIENTIST, UserRoleTypes.DOCTOR, UserRoleTypes.ADMIN, UserRoleTypes.EDIFIER, UserRoleTypes.VIEWER], userRoles)) {
            const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
            return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
        }
    
        if (!await this.verifyPassword(user.password, password)) {
            const errorMessage = getErrorMessage(HttpErrors.PASSWORD_MISMATCH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        const jwtId = uuidv4();
        const token = generateSignToken({userId: user.id, roles: userRoles, jwtId} as Token);
        const refreshToken = await generateRefreshToken(user, jwtId);    
        return res.send({token, refreshTokenId: refreshToken.id});
    }

    public async logout (req: Request, res: Response) {
        const token = req.headers['authorization'];
        res.removeHeader('token');
        res.removeHeader('authorization');
        if (!token) {
            const errorMessage = getErrorMessage(HttpErrors.NO_TOKEN);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const decodedToken = verifyToken(token);
        if (decodedToken === undefined) {
            const errorMessage = getErrorMessage(HttpErrors.TOKEN_VERIFICATION_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const refreshTokenRepo = getCustomRepository(RefreshTokenRepository);
        let refreshToken: RefreshToken;    
        try {
            refreshToken = await refreshTokenRepo.findByJwtIdOrFail(decodedToken.jwtId);
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
            return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
        }
        await refreshTokenRepo.remove(refreshToken);        
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    }

    public async refreshToken (req: Request, res: Response) {
        const token = req.headers['authorization'];
        const {refreshTokenId} = req.body;
        if (!token) {
            const errorMessage = getErrorMessage(HttpErrors.NO_TOKEN);
            return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
        }
        if (!refreshTokenId) {
            const errorMessage = getErrorMessage(HttpErrors.NO_REFRESH_TOKEN);
            return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
        }
        const decodedToken = verifyToken(token, true, true);
        if (decodedToken === undefined) {
            const errorMessage = getErrorMessage(HttpErrors.TOKEN_VERIFICATION_ERROR);
            return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
        }

        const decodedRefreshToken = await verifyRefreshToken(refreshTokenId, decodedToken);
        if (decodedRefreshToken === undefined) {
            const errorMessage = getErrorMessage(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
            return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
        }
        const connection = getConnection();    
        const refreshTokenRepo = connection.getRepository(RefreshToken);
        await refreshTokenRepo.remove(decodedRefreshToken);

        const usersRepository = connection.getCustomRepository(UserRepository);
        let user: User;    
        try {
            user = await usersRepository.findActiveByIdOrFail(decodedToken.userId)
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const jwtId = uuidv4();
        const userRoles = user.roles?.map((roleEntity) => roleEntity.role);
        const newToken = generateSignToken({userId: user.id, roles: userRoles, jwtId} as Token);
        const newRefreshToken = await generateRefreshToken(user, jwtId);
        return res.send({token: newToken, refreshTokenId: newRefreshToken.id});
    }
    
    public async registerUser (req: Request, res: Response) {
        if (!(req.body.login && req.body.password)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        let requestBody: RegisterUserBody;
        try {
            requestBody = {
                email: req.body.email,
                login: req.body.login,
                password: req.body.password,
                phoneNumber: req.body.phone_number,
                comment: req.body.comment,
                roles: req.body.roles,
                patients: req.body.patients
            };
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const userRepo = getCustomRepository(UserRepository);
        const rolesRepo = getCustomRepository(RolesRepository);

        const existingUser = await userRepo.findByLogin(requestBody.login);
        if (existingUser != undefined) {
            const errorMessage = getErrorMessage(HttpErrors.LOGIN_TAKEN);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const myRoles: Array<Roles> = [];
        if (requestBody.roles != undefined) {
            for (const role of requestBody.roles) {
                try {
                    const getRole = await rolesRepo.findByStringOrFail(role);
                    myRoles.push(getRole);
                } catch (error) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_ROLE);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
            }
        }

        let myPatients: Array<User> = [];
        if (requestBody.patients != undefined) {
            if (!(requestBody.roles?.includes(UserRoleTypes.EDIFIER) || 
                    requestBody.roles?.includes(UserRoleTypes.VIEWER) || 
                    requestBody.roles?.includes(UserRoleTypes.DOCTOR) || 
                    requestBody.roles?.includes(UserRoleTypes.DATA_SCIENTIST))) {
                const errorMessage = getErrorMessage(HttpErrors.NO_PATIENTS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            } 
            myPatients = await userRepo.find({where: {login: In(requestBody.patients)}});
            if (myPatients.length != requestBody.patients.length) {
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        const user = new User();
        user.login = UserLogin.create(requestBody.login).getLogin();
        user.email = requestBody.email?.toLowerCase();
        user.password = await argon2.hash(requestBody.password);
        user.phone_number = requestBody.phoneNumber;
        user.comment = requestBody.comment;
        if (myRoles.length > 0) {
            user.roles = myRoles;
        } else {
            const patientRole = await rolesRepo.findByStringOrFail(UserRoleTypes.PATIENT);
            user.roles = [patientRole];
        }

        if (requestBody.patients != undefined) {
            const myDoctorsPatients: Array<DoctorsPatients> = [];
            for (const patient of myPatients) {
                const doctorsPatient = new DoctorsPatients;
                doctorsPatient.doctor = user;
                doctorsPatient.patient = patient;
                myDoctorsPatients.push(doctorsPatient);
            }
            user.patients = myDoctorsPatients;
            user.is_all_patients = false;
        } 
        
        try {
            await userRepo.save(user);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async selfRegister (req: Request, res: Response) {
        if (!req.body.login) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const login = String(req.body.login);

        const userRepo = getCustomRepository(UserRepository);
        const existingUser = await userRepo.findByEmailOrNumber(login.toLowerCase());
        if (existingUser != undefined) {
            if (existingUser.password == undefined) {
                try {
                    const lang = req.query.lang as string ?? 'ru';
                    await this.generateAndSendCode(existingUser, lang);
                    return res.status(HttpStatusCodes.SUCCESS).send({
                        user_id: existingUser.id,
                        login: existingUser.login, 
                        email: existingUser.email, 
                        phone_number: existingUser.phone_number
                    });
                } catch (error) {
                    console.error(error);
                    const errorMessage = getErrorMessage(HttpErrors.MESSAGE_SENDING_ERROR);
                    return res.status(HttpStatusCodes.ERROR).send(errorMessage);
                }
            }
            const errorMessage = getErrorMessage(HttpErrors.LOGIN_TAKEN);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        
        let email: string;
        let phoneNumber: string;
        if (this.regexpEmail.test(login.toLowerCase())) {
            email = login.toLowerCase();
        } else if (this.regexpNumber.test(login)) {
            phoneNumber = login;
            console.log(`Number to validate: ${phoneNumber}`);
            const isValid = await this.validatePhoneNumber(phoneNumber);
            if (!isValid) {
                const errorMessage = getErrorMessage(HttpErrors.INVALID_NUMBER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            } 
        } else {
            const errorMessage = getErrorMessage(HttpErrors.WRONG_FORMAT);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const rolesRepo = getCustomRepository(RolesRepository);
        const patientRole = await rolesRepo.findByStringOrFail(UserRoleTypes.PATIENT);
        const myRoles: Array<Roles> = [];
        if (req.body.roles != undefined) {
            for (const role of req.body.roles) {
                if (role != patientRole.role) {
                    const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
                    return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
                    // TODO: admin notification
                }
                try {
                    const getRole = await rolesRepo.findByStringOrFail(role);
                    myRoles.push(getRole);
                } catch (error) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_ROLE);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
            }
        }

        const user = new User();
        user.login = email ?? phoneNumber;
        user.email = email;
        user.phone_number = phoneNumber;
        user.is_active = false;
        if (myRoles.length > 0) {
            user.roles = myRoles;
        } else {
            user.roles = [patientRole];
        }

        try {
            await userRepo.save(user); 
            const lang = req.query.lang as string ?? 'ru';
            await this.generateAndSendCode(user, lang);
            return res.status(HttpStatusCodes.SUCCESS).send({
                user_id: user.id, 
                login: user.login, 
                email: user.email, 
                phone_number: user.phone_number
            });
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.MESSAGE_SENDING_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async sendCode (req: Request, res: Response) {
        const login = String(req.body.login).toLowerCase();

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(login);
        if (user == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!this.regexpEmail.test(login) && !this.regexpNumber.test(login)) {
            const errorMessage = getErrorMessage(HttpErrors.WRONG_FORMAT);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
            
        try {
            const lang = req.query.lang as string ?? 'ru';
            await this.generateAndSendCode(user, lang);
            return res.status(HttpStatusCodes.SUCCESS).send({
                user_id: user.id,
                login: user.login, 
                email: user.email, 
                phone_number: user.phone_number
            });
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.MESSAGE_SENDING_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async verifyCode (req: Request, res: Response) {
        const user_id = Number(req.params.id);
        const verificationCode = Number(req.body.verification_code);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findOne(user_id);
        if (user == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        try { 
            const verificationRepo = getCustomRepository(VerificationCodesRepository);
            const existingCode = await verificationRepo.findCodeByUserId(user.id); 
            const isCodeExpired: boolean = moment().isAfter(existingCode.date_expired);
            if (verificationCode == existingCode.code && !isCodeExpired) {
                existingCode.is_verified = true;
                await verificationRepo.save(existingCode);
            } else {
                const errorMessage = getErrorMessage(HttpErrors.WRONG_VERIFICATION_CODE);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }    
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async setPassword (req: Request, res: Response) {
        if (!(req.body.password && req.body.repeat_password)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const user_id = Number(req.params.id);
        const password = String(req.body.password);
        const repeatPassword = String(req.body.repeat_password);
        if (password != repeatPassword) {
            const errorMessage = getErrorMessage(HttpErrors.PASSWORD_MISMATCH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findOne(user_id);
        if (user == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const verificationCode = await verificationRepo.findCodeByUserId(user.id); 
        if (!verificationCode.is_verified) {
            const errorMessage = getErrorMessage(HttpErrors.WRONG_VERIFICATION_CODE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        } 

        try {
            user.password = await argon2.hash(password);
            if (!user.is_active) {
                user.is_active = true;
            }
            await userRepo.save(user);
            verificationCode.is_verified = false;
            await verificationRepo.save(verificationCode);
            const refreshTokenRepo = getCustomRepository(RefreshTokenRepository);
            const refreshToken = await refreshTokenRepo.findActiveByUserId(user.id);
            if (refreshToken != undefined) {
                await refreshTokenRepo.remove(refreshToken); 
            }
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async changePassword (req: Request, res: Response) {
        if (!(req.body.password && req.body.repeat_password)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const userId = Number(req.token.userId);
        const password = String(req.body.password);
        const repeatPassword = String(req.body.repeat_password);
        if (password != repeatPassword) {
            const errorMessage = getErrorMessage(HttpErrors.PASSWORD_MISMATCH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findOne(userId);
        user.password = await argon2.hash(password);

        try {
            await userRepo.save(user);
         
            const refreshTokenRepo = getCustomRepository(RefreshTokenRepository);
            const refreshToken = await refreshTokenRepo.findActiveByUserId(user.id);
            if (refreshToken != undefined) {
                await refreshTokenRepo.remove(refreshToken); 
            }
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async patchUser (req: Request, res: Response) {
        let requestBody: PatchUserBody;
        const userId = Number(req.params.id);

        try {
            requestBody = {
                password: req.body.password,
                phoneNumber: req.body.phone_number,
                comment: req.body.comment,
                roles: req.body.roles,
                patients: req.body.patients,
                isActive: req.body.is_active,
                isCheckHealthy: req.body.is_check_healthy,
                isCheckCovid: req.body.is_check_covid,
                isValidateCough: req.body.is_validate_cough,
                checkStart: req.body.check_start,
                checkEnd: req.body.check_end
            };
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const userRepo = getCustomRepository(UserRepository);
        const rolesRepo = getCustomRepository(RolesRepository);

        const user = await userRepo.findOne(userId);
        if (user == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const regexpTime = new RegExp('^[0-2][0-9]:[0-5][0-9]$');
        if ((requestBody.checkStart && (!regexpTime.test(requestBody.checkStart) || !requestBody.checkEnd)) ||
            (requestBody.checkEnd && (!regexpTime.test(requestBody.checkEnd) || !requestBody.checkStart))) {
                const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const myRoles: Array<Roles> = [];
        if (requestBody.roles != undefined) {
            for (const role of requestBody.roles) {
                try {
                    const getRole = await rolesRepo.findByStringOrFail(role);
                    myRoles.push(getRole);
                } catch (error) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_ROLE);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
            }
            user.roles = myRoles;
        }

        if (requestBody.patients != undefined) {
            const userRolesList = await getUserRoles(user.id);
            if (!(userRolesList.includes(UserRoleTypes.EDIFIER) ||
                    userRolesList.includes(UserRoleTypes.VIEWER) ||  
                    userRolesList.includes(UserRoleTypes.DOCTOR) || 
                    userRolesList.includes(UserRoleTypes.DATA_SCIENTIST))) {
                const errorMessage = getErrorMessage(HttpErrors.NO_PATIENTS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            
            const myPatients = await userRepo.find({where: {login: In(requestBody.patients)}});
            if (myPatients.length != requestBody.patients.length) {
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            const myDoctorsPatients: Array<DoctorsPatients> = [];
            for (const patient of myPatients) {
                const doctorsPatient = new DoctorsPatients;
                doctorsPatient.doctor = user;
                doctorsPatient.patient = patient;
                myDoctorsPatients.push(doctorsPatient);
            }
            user.patients = myDoctorsPatients;
            user.is_all_patients = false;
        }

        if (requestBody.password != undefined) {
            user.password = await argon2.hash(requestBody.password);
        }
        user.phone_number = requestBody?.phoneNumber ?? user.phone_number;
        user.comment = requestBody?.comment ?? user.comment;
        user.is_active = requestBody?.isActive ?? user.is_active;
        user.is_check_healthy = requestBody?.isCheckHealthy ?? user.is_check_healthy;   
        user.is_check_covid = requestBody?.isCheckCovid ?? user.is_check_covid; 
        user.is_validate_cough = requestBody?.isValidateCough ?? user.is_validate_cough; 
        user.check_start = requestBody?.checkStart ?? user.check_start;
        user.check_end = requestBody?.checkEnd ?? user.check_end;

        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            if (requestBody.patients != undefined) {
                await manager.delete(DoctorsPatients, {doctor_id: user.id});
            }
            await manager.save(user);
            await queryRunner.commitTransaction()
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        } finally {
            await queryRunner.release();
        }
    }

    public async getPersonalData (req: Request, res: Response) {
        const userId = Number(req.token.userId);
        const connection = getConnection();
        const personalDataRepo = connection.getRepository(PersonalData);
        const userData = await personalDataRepo.findOne({where: {user_id: userId}});
        if (userData == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_PERSONAL_DATA);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
       
        const response = {
            identifier: userData.identifier,
            age: userData.age,
            gender: userData.gender?.gender_type,
            is_smoking: userData.is_smoking,
            voice_audio_path: userData.voice_audio_path
        }
        try {
            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async patchPersonalData (req: Request, res: Response) {
        let requestBody: PatchDataBody;
        const userId = Number(req.token.userId);

        try {
            requestBody = {
                identifier: req.body.identifier,
                age: req.body.age,
                gender: req.body.gender,
                is_smoking: req.body.is_smoking
            };
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const connection = getConnection();
        const personalDataRepo = connection.getRepository(PersonalData);
        let userData: PersonalData;
        userData = await personalDataRepo.findOne({where: {user_id: userId}});
        if (userData == undefined) {
            userData = new PersonalData();
            userData.user_id = userId;
        }

        userData.identifier = requestBody?.identifier ?? userData.identifier;
        userData.age = requestBody?.age ?? userData.age;
        userData.is_smoking = requestBody?.is_smoking ?? userData.is_smoking;
        if (requestBody.gender) {
            const gender = await getCustomRepository(GenderTypesRepository).findByStringOrFail(requestBody.gender);
            userData.gender = gender;
        }

        try {
            await personalDataRepo.save(userData);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async dischargePersonalData (req: Request, res: Response) {
        const userId = Number(req.params.id);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;

        try {
            const user = await manager.findOne(User, {where: {id: userId}});
            if (user == undefined) {
                await queryRunner.rollbackTransaction();
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            const userData = await manager.findOne(PersonalData, {where: {user_id: userId}});
            if (userData == undefined) {
                await queryRunner.rollbackTransaction();
                const errorMessage = getErrorMessage(HttpErrors.NO_PERSONAL_DATA);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }

            if (userData.voice_audio_path != undefined) {
                await fileService.deleteDirectory(userData.voice_audio_path);
            }
            await manager.delete(PersonalData, {user_id: userId});
            user.is_active = true;
            await manager.save(user, {transaction: false});
            await queryRunner.commitTransaction();

            const message = 'Your personal data is discharged successfully, you can now use the Acoustery app!';
            if (user.phone_number != undefined) {
                await notificationSmsService.sendUserSms(user.phone_number, message);
            }
            if (user.email != undefined) {
                if (user.is_email_error) {
                    console.error(`Can not send message to this email due to ${user.email_error_type}`);
                    const errorMessage = getErrorMessage(HttpErrors.MESSAGE_SENDING_ERROR);
                    return res.status(HttpStatusCodes.ERROR).send(errorMessage);
                }
                await notificationEmailService.sendUserEmail(user.email, message, 'Your data is discharged!');
            }
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            await queryRunner.rollbackTransaction();
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        } finally {
            await queryRunner.release();
        }
    }

    public async deleteUser (req: Request, res: Response) {
        const userId = Number(req.params.id);
        
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const user = await manager.findOne(User, {where: {id: userId}});
            if (user == undefined) {
                await queryRunner.rollbackTransaction();
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            if (user.password != undefined) {
                user.is_active = false;
                user.login = `deleted (${user.login})`;
                user.email = user.email ? `deleted (${user.email})` : undefined;
                user.phone_number = user.phone_number ? `deleted (${user.phone_number})` : undefined;
                await manager.save(user);
            } else {
                await manager.delete(VerificationCodes, {user_id: userId});
                await manager.delete(User, {id: userId});
            }
            
            await queryRunner.commitTransaction();
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch (error) {
            console.error(error);
            await queryRunner.rollbackTransaction();
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        } finally {
            await queryRunner.release();
        }
    }
}
