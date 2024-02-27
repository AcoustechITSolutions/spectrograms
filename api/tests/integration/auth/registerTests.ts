import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection, getCustomRepository} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';
import {HttpErrors, errorDescription} from '../../../src/helpers/status';
import {createTestPatient, DEFAULT_PASSWORD} from '../stubs';
import {VerificationCodesRepository} from '../../../src/infrastructure/repositories/verificationCodesRepo';
import {UserRepository} from '../../../src/infrastructure/repositories/userRepo';
import {Server} from 'http';
import {seedDb} from '../../../src/infrastructure/seed';
import {delay} from '../helper';
import config from '../../../src/config/config';
import {notificationSns} from '../../../src/container';

let connection: Connection;
const app: Application = express();
let serverConnection: Server;

describe('Self registration tests', function() {
    this.timeout(100000);
    beforeEach(async () => {
        return new Promise(async (resolve, reject) => {
            await setupServer(app);
            connection = await setupDatabaseConnection(false, true);
            await seedDb(connection);
            serverConnection = app.listen(PORT, () => {
                console.log(`server running at port: ${PORT}`);
                resolve();
            });
        });
    });

    afterEach(async () => {
        return new Promise(async (resolve, reject) => {
            await connection.dropDatabase();
            await connection.close();
            await serverConnection.close(reject);
            resolve();
        });
    });

    it('1.1 Can self register with email', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(process.env.TEST_EMAIL);
        expect(user.email).to.eq(process.env.TEST_EMAIL);
        expect(user.password).to.be.null;
        expect(user.phone_number).to.be.null;
        expect(user.is_active).to.eq(false);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');
    });

    it('1.2 Can self register with phone number', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_NUMBER
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_NUMBER);
        expect(res.body.email).to.be.null;
        expect(res.body.phone_number).to.eq(process.env.TEST_NUMBER);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(process.env.TEST_NUMBER);
        expect(user.email).to.be.null;
        expect(user.password).to.be.null;
        expect(user.phone_number).to.eq(process.env.TEST_NUMBER);
        expect(user.is_active).to.eq(false);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');
    });
  
    it('1.3 Cannot self register with wrong format email', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': 'bademail'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.WRONG_FORMAT);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_FORMAT));
    });

    it('1.4 Cannot self register with wrong format number', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': '89169897032'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.WRONG_FORMAT);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_FORMAT));
    });

    it('1.5 Cannot self register with invalid email', async () => {
        let subscriptionArn: string;
        const snsSubscriptionParams = {
            TopicArn: process.env.BOUNCE_TOPIC_ARN,
            Protocol: 'https',
            Endpoint: `${process.env.API_ENDPOINT}/sns/ses_bounces`,
            ReturnSubscriptionArn: true,
            Attributes: {
                DeliveryPolicy: '{"healthyRetryPolicy": {"minDelayTarget": 1, "maxDelayTarget": 1, "numRetries": 0}}' 
            }
        };
        notificationSns.subscribe(snsSubscriptionParams, function(err, data) {
            expect(err).to.be.null;
            console.log(data);
            subscriptionArn = data.SubscriptionArn;
        });
        await delay('30s');

        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.BAD_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.BAD_EMAIL);
        expect(res.body.email).to.eq(process.env.BAD_EMAIL);
        expect(res.body.phone_number).to.be.null;

        await delay('30s');
        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(process.env.BAD_EMAIL);
        expect(user.email).to.eq(process.env.BAD_EMAIL);
        expect(user.password).to.be.null;
        expect(user.phone_number).to.be.null;
        expect(user.is_active).to.eq(false);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');
        expect(user.is_email_error).to.eq(true);
        expect(user.email_error_type).to.eq('Bounce');

        const resendResponse = await supertest(app)
            .post('/v1.2/user/send_code')
            .send({
                'login': process.env.BAD_EMAIL
            })
            .expect(500)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(resendResponse.body.status).to.eq(HttpErrors.MESSAGE_SENDING_ERROR);
        expect(resendResponse.body.error).to.eq(errorDescription.get(HttpErrors.MESSAGE_SENDING_ERROR));

        const unsubscribeParams = {
            SubscriptionArn: subscriptionArn
        };
        notificationSns.unsubscribe(unsubscribeParams, function(err, data) {
            expect(err).to.be.null;
            console.log(data);
        })
        await delay('10s');
    });

    it('1.6 Cannot self register with invalid number', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': '+12345'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INVALID_NUMBER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INVALID_NUMBER));
    });

    it('1.7 Cannot self register with wrong data', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'email': process.env.TEST_EMAIL
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });

    it('1.8 Cannot self register if login is taken', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': user.email
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.LOGIN_TAKEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.LOGIN_TAKEN));
    });
   
    it('1.9 Can self register if login is taken but password is not set', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(process.env.TEST_EMAIL);
        expect(user.email).to.eq(process.env.TEST_EMAIL);
        expect(user.password).to.be.null;
        expect(user.phone_number).to.be.null;
        expect(user.is_active).to.eq(false);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');

        const secondResponse = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(secondResponse.body.user_id).to.not.be.null;
        expect(secondResponse.body.login).to.eq(process.env.TEST_EMAIL);
        expect(secondResponse.body.email).to.eq(process.env.TEST_EMAIL);
        expect(secondResponse.body.phone_number).to.be.null;
    });

    it('2.1 Can resend code to email', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const firstCode = await verificationRepo.findCodeByUserId(userId);
        expect(firstCode).to.not.be.null; 

        const resendResponse = await supertest(app)
            .post('/v1.2/user/send_code')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(resendResponse.body.user_id).to.not.be.null;
        expect(resendResponse.body.login).to.eq(process.env.TEST_EMAIL);
        expect(resendResponse.body.email).to.eq(process.env.TEST_EMAIL);
        expect(resendResponse.body.phone_number).to.be.null;

        const secondCode = await verificationRepo.findCodeByUserId(userId);
        expect(secondCode.code).to.not.eq(firstCode.code);
    });

    it('2.2 Can resend code to phone number', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_NUMBER
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_NUMBER);
        expect(res.body.email).to.be.null;
        expect(res.body.phone_number).to.eq(process.env.TEST_NUMBER);

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const firstCode = await verificationRepo.findCodeByUserId(userId);
        expect(firstCode).to.not.be.null; 

        const resendResponse = await supertest(app)
            .post('/v1.2/user/send_code')
            .send({
                'login': process.env.TEST_NUMBER
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(resendResponse.body.user_id).to.not.be.null;
        expect(resendResponse.body.login).to.eq(process.env.TEST_NUMBER);
        expect(resendResponse.body.email).to.be.null;
        expect(resendResponse.body.phone_number).to.eq(process.env.TEST_NUMBER);

        const secondCode = await verificationRepo.findCodeByUserId(userId);
        expect(secondCode.code).to.not.eq(firstCode.code);
    });
  
    it('2.3 Cannot send code if user does not exist', async () => {
        const res = await supertest(app)
            .post('/v1.2/user/send_code')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('2.4 Cannot send code to wrong format login', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/user/send_code')
            .send({
                'login': user.login
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.WRONG_FORMAT);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_FORMAT));
    });

    it('3.1 Can verify code', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        expect(code.is_verified).to.eq(false);
        await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code
            })
            .expect(204);

        const verifiedCode = await verificationRepo.findCodeByUserId(userId); 
        expect(verifiedCode.is_verified).to.eq(true);
    });

    it('3.2 Cannot verify code if user does not exist', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const maxId = await userRepo.findMaxId() ?? 0;
        const res = await supertest(app)
            .post(`/v1.2/user/${maxId + 1}/verify`)
            .send({
                'verification_code': '1234'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('3.3 Cannot verify wrong code', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        const verifyResponse = await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code + 1
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(verifyResponse.body.status).to.eq(HttpErrors.WRONG_VERIFICATION_CODE);
        expect(verifyResponse.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_VERIFICATION_CODE));
    });

    it('3.4 Cannot verify expired code', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        await delay(config.verificationCodeLife);
        const verifyResponse = await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(verifyResponse.body.status).to.eq(HttpErrors.WRONG_VERIFICATION_CODE);
        expect(verifyResponse.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_VERIFICATION_CODE));
    });

    it('4.1 Can set password', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code
            })
            .expect(204);

        await supertest(app)
            .post(`/v1.2/user/${userId}/password`)
            .send({
                'password': DEFAULT_PASSWORD,
                'repeat_password': DEFAULT_PASSWORD
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(process.env.TEST_EMAIL);
        expect(user.password).to.not.be.null;
        expect(user.is_active).to.eq(true);

        const loginResponse = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': process.env.TEST_EMAIL,
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(loginResponse.body.token).to.not.be.null;
        expect(loginResponse.body.refreshTokenId).to.not.be.null;
    });

    it('4.2 Cannot set password with not enough data', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code
            })
            .expect(204);

        const passwordResponse = await supertest(app)
            .post(`/v1.2/user/${userId}/password`)
            .send({
                'password': DEFAULT_PASSWORD
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(passwordResponse.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(passwordResponse.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });
    
    it('4.3 Cannot set password if repeated incorrectly', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        await supertest(app)
            .post(`/v1.2/user/${userId}/verify`)
            .send({
                'verification_code': code.code
            })
            .expect(204);

        const passwordResponse = await supertest(app)
            .post(`/v1.2/user/${userId}/password`)
            .send({
                'password': DEFAULT_PASSWORD,
                'repeat_password': 'badpassword'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(passwordResponse.body.status).to.eq(HttpErrors.PASSWORD_MISMATCH);
        expect(passwordResponse.body.error).to.eq(errorDescription.get(HttpErrors.PASSWORD_MISMATCH));
    });

    it('4.4 Cannot set password if user does not exist', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const maxId = await userRepo.findMaxId() ?? 0;
        const res = await supertest(app)
            .post(`/v1.2/user/${maxId + 1}/password`)
            .send({
                'password': DEFAULT_PASSWORD,
                'repeat_password': DEFAULT_PASSWORD
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });
    
    it('4.5 Cannot set password if code is not verified', async () => {
        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        expect(res.body.login).to.eq(process.env.TEST_EMAIL);
        expect(res.body.email).to.eq(process.env.TEST_EMAIL);
        expect(res.body.phone_number).to.be.null;

        const userId = res.body.user_id;
        const verificationRepo = getCustomRepository(VerificationCodesRepository);
        const code = await verificationRepo.findCodeByUserId(userId); 
        expect(code.is_verified).to.eq(false);
        const passwordResponse = await supertest(app)
            .post(`/v1.2/user/${userId}/password`)
            .send({
                'password': DEFAULT_PASSWORD,
                'repeat_password': DEFAULT_PASSWORD
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(passwordResponse.body.status).to.eq(HttpErrors.WRONG_VERIFICATION_CODE);
        expect(passwordResponse.body.error).to.eq(errorDescription.get(HttpErrors.WRONG_VERIFICATION_CODE));
    });
});