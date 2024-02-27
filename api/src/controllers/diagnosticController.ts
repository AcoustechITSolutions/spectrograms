import {Request, Response} from 'express';
import config from '../config/config';

import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {DiagnosticRequest} from '../infrastructure/entity/DiagnosticRequest';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus} from '../domain/RequestStatus';
import {getRepository, getConnection, getManager, getCustomRepository} from 'typeorm';
import {CoughAudio} from '../infrastructure/entity/CoughAudio';
import {PatientInfo} from '../infrastructure/entity/PatientInfo';
import {PersonalData} from '../infrastructure/entity/PersonalData';
import {Gender} from '../domain/Gender';
import {isValidFileExt} from '../helpers/file';
import {UploadedFile} from 'express-fileupload';
import {DiagnosticReport} from '../infrastructure/entity/DiagnosticReport';
import {CoughCharacteristics} from '../infrastructure/entity/CoughCharacteristic';

import {fileService} from '../container';
import {DiagnosticRequestStatus as EntityDiagnosticRequestStatus} from '../infrastructure/entity/DiagnostRequestStatus';
import {DiagnosticRequestStatusRepository} from '../infrastructure/repositories/diagnosticRequestStatusRepo';

import {inferenceDiagnostic} from '../services/diagnostic/diagnosticInferenceService';
import {DiagnosticRequestRepository} from '../infrastructure/repositories/diagnosticRequestRepo';
import {DiagnosticCoughAudioRepository} from '../infrastructure/repositories/diagnosticCoughAudioRepo';

import {generateDiagnosticHL7} from '../services/hl7Service';
import {fetchDiagnostic, fetchDiagnosticById, fetchDiagnosticByQR, fetchSpectrogramPath, fetchHL7InfoById} from '../services/query/diagnosticQueryService';

export type UserReportsResponse = {
    status: string,
    date: Date,
    request_id: number,
    identifier?: string,
    qr_code_url?: string,
    report?: UserDiagnosticReport
}

export type UserDiagnosticReport = {
    probability?: number | string,
    diagnosis?: string,
    productivity?: string,
    intensity?: string,
    commentary?: string
}

type PatchCoughAudioRequest = {
    cough_audio: UploadedFile
}

type GetDiagnosticByIdResponse = {
    status: string,
    date: Date,
    request_id: number,
    age?: number,
    gender?: string,
    identifier?: string,
    probability?: number,
    diagnosis?: string,
    productivity?: string,
    intensity?: string
}

type DiagnosticBody = {
    is_forced?: boolean,
    sick_days?: number,
    age?: number,
    gender?: Gender,
    identifier?: string,
    is_smoking?: boolean,
    location?: {
        latitude: number,
        longitude: number
    }
}

const validExtensions = ['.wav'];
const validMIME = ['audio/wav', 'audio/x-wav', 'audio/vnd.wave', 'audio/wave'];

/**
 * Handles user request to /diagnostic route.
 * For admin processing of this record see ProcessingController.
 */
export class DiagnosticController { // TODO: add dependencies through di

    public async diagnosticNew (req: Request, res: Response) {
        try {
            const audio = req.files.cough_audio as UploadedFile;
            const photo = req.files.patient_photo as UploadedFile;
            const userId = req.token.userId;
    
            if (!isValidFileExt(validExtensions, audio.name)) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }

            let requestBody: DiagnosticBody;
            try {
                requestBody = {
                    is_forced: JSON.parse(req.body?.is_force ?? null),
                    sick_days: req.body?.sick_days,
                    identifier: req.body?.identifier,
                    age: req.body?.age,
                    gender: req.body?.gender,
                    is_smoking: JSON.parse(req.body?.is_smoking ?? null),
                    location: {
                        latitude: req.body?.location_latitude,
                        longitude: req.body?.location_longitude
                    }
                };
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
    
            const connection = getConnection();
            const personalDataRepo = connection.getRepository(PersonalData); 
            const userData = await personalDataRepo.findOne({where: {user_id: userId}});
    
            const diagnosticRequest = new DiagnosticRequest();
            diagnosticRequest.user_id = userId;
            diagnosticRequest.language = req.language; 
            diagnosticRequest.location_latitude = requestBody.location?.latitude;
            diagnosticRequest.location_longitude = requestBody.location?.longitude;
    
            const statusRepo = connection.getCustomRepository(DiagnosticRequestStatusRepository);
            let processingStatus: EntityDiagnosticRequestStatus;
            try {
                processingStatus = await statusRepo.findByStringOrFail(DomainDiagnosticRequestStatus.PROCESSING);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }
            diagnosticRequest.status_id = processingStatus.id;

            const patientInfo = new PatientInfo();
            if (userData && (!requestBody.identifier && !requestBody.age && !requestBody.gender && requestBody.is_smoking == undefined)) {
                patientInfo.identifier = userData.identifier;
                patientInfo.age = userData.age;
                patientInfo.gender = userData.gender?.gender_type;
                patientInfo.is_smoking = userData.is_smoking;
            } else {
                patientInfo.identifier = requestBody.identifier;
                patientInfo.age = requestBody.age;
                patientInfo.gender = requestBody.gender;
                patientInfo.is_smoking = requestBody.is_smoking;
            }
            patientInfo.sick_days = requestBody.sick_days;
            patientInfo.request = diagnosticRequest;
    
            const coughCharacteristic = new CoughCharacteristics();
            coughCharacteristic.request = diagnosticRequest;
            coughCharacteristic.is_forced = requestBody.is_forced;
    
            const queryRunner = connection.createQueryRunner();
            await queryRunner.startTransaction();
            const manager = queryRunner.manager;
            let coughAudio: CoughAudio;
            let audio_path: string;
            let diagnosticReport: DiagnosticReport;
            try {
                await manager.save([diagnosticRequest, patientInfo, coughCharacteristic], {
                    transaction: false,
                });

                if (photo != undefined) {
                    const photo_path = `${config.photoFolder}/${userId}/${diagnosticRequest.id}/${photo.name}`;
                    patientInfo.photo_path = await fileService.saveFile(photo_path, photo.data);
                    await manager.save(patientInfo, {transaction: false});
                }
                coughAudio = new CoughAudio();
                audio_path = `${config.audioFolder}/${userId}/${diagnosticRequest.id}/${audio.name}`;
                coughAudio.request = diagnosticRequest;
                coughAudio.file_path = await fileService.saveFile(audio_path, audio.data);
    
                diagnosticReport = new DiagnosticReport();
                diagnosticReport.user_id = userId;
                diagnosticReport.request = diagnosticRequest;
                await manager.save([coughAudio, diagnosticReport], {transaction: false});
                await queryRunner.commitTransaction();
            } catch (error) {
                console.error(error);
                await queryRunner.rollbackTransaction();
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            } finally {
                await queryRunner.release();
            }
    
            inferenceDiagnostic(coughAudio.file_path, diagnosticRequest.id);
            return res.status(HttpStatusCodes.CREATED).send({'request_id': diagnosticRequest.id});
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.ERROR).send();
        }
    }
    
    // TODO: delete after satellite server is done
    public async getDiagnostic (req: Request, res: Response) {
        const userId = req.token.userId;
        const paginationParams = req.paginationParams;
    
        try {
            const userReportsResult = await fetchDiagnostic(userId, paginationParams);     
    
            const response: UserReportsResponse[] = userReportsResult?.map((userReport) => {
                const status = userReport.status;
                const date = userReport.date;
                const request_id = userReport.request_id;
                const qr_token = userReport.qr_code_token;
                const qr_code_url = `${process.env.QR_URL}/${request_id}/?token=${qr_token}`;
                const report = status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    probability: userReport.probability ?? '',
                    diagnosis: userReport.diagnosis ?? '',
                    productivity: userReport.productivity ?? '',
                    intensity: userReport.intensity ?? '',
                    commentary: userReport.commentary ?? '',
                } : undefined;
                return {status, date, request_id, qr_code_url, report};
            });
    
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }

    public async getDiagnosticV1_2 (req: Request, res: Response) {
        const userId = req.token.userId;
        const paginationParams = req.paginationParams;
    
        try {
            const userReportsResult = await fetchDiagnostic(userId, paginationParams);     
    
            const response: UserReportsResponse[] = userReportsResult?.map((userReport) => {
                const identifier = userReport.identifier;
                const status = userReport.status;
                const date = userReport.date;
                const request_id = userReport.request_id;
                const qr_token = userReport.qr_code_token;
                const qr_code_url = `${process.env.QR_URL}/${request_id}/?token=${qr_token}`;
                const report = status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    probability: userReport.probability,
                    diagnosis: userReport.diagnosis,
                    productivity: userReport.productivity,
                    intensity: userReport.intensity
                } : undefined;
                return {status, date, request_id, identifier, qr_code_url, report};
            });
    
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }

    public async getDiagnosticById(req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = Number(req.token.userId);

        try {
            const dbResponse = await fetchDiagnosticById(userId, requestId);
            if (dbResponse == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }
            const qrToken = dbResponse.qr_code_token;
            const qrCodeUrl = `${process.env.QR_URL}?id=${requestId}&token=${qrToken}`;

            const response: UserReportsResponse = {
                status: dbResponse.status,
                date: dbResponse.date,
                request_id: dbResponse.request_id,
                qr_code_url: qrCodeUrl,
                report: dbResponse.status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    probability: dbResponse.probability,
                    diagnosis: dbResponse.diagnosis
                } : undefined
            };
            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getDiagnosticByIdV1_2(req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = Number(req.token.userId);

        try {
            const dbResponse = await fetchDiagnosticById(userId, requestId);
            if (dbResponse == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }
            const qrToken = dbResponse.qr_code_token;
            const qrCodeUrl = `${process.env.QR_URL}?id=${requestId}&token=${qrToken}`;

            const response: UserReportsResponse = {
                status: dbResponse.status,
                date: dbResponse.date,
                request_id: dbResponse.request_id,
                identifier: dbResponse.identifier,
                qr_code_url: qrCodeUrl,
                report: dbResponse.status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    probability: dbResponse.probability,
                    diagnosis: dbResponse.diagnosis,
                    productivity: dbResponse.productivity,
                    intensity: dbResponse.intensity
                } : undefined
            };
            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getDiagnosticSpectrogram (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = Number(req.token.userId);
    
        const spectrogram = await fetchSpectrogramPath(userId, requestId);
        if (spectrogram == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        res.status(HttpStatusCodes.SUCCESS);
        res.setHeader('Content-Type', 'image/png');
        try {
            const stream = await fileService
                .getFileAsStream(spectrogram);
         
            stream.on('error', (err) => {
                console.error(err);
                const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            })
                .pipe(res);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getDiagnosticHL7 (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = Number(req.token.userId);

        const requestInfo = await fetchHL7InfoById(userId, requestId);
        if (requestInfo == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        let fullPath: string;
        try {
            const hl7 = generateDiagnosticHL7({
                requestId: requestId,
                dateCreated: requestInfo.date,
                status: requestInfo.status,
                diagnosis: requestInfo.diagnosis,
                probability: requestInfo.probability,
                commentary: requestInfo.commentary,
                userId: requestInfo.user_id,
                userLogin: requestInfo.user_login,
                age: requestInfo.age,
                gender: requestInfo.gender,
                identifier: requestInfo.identifier,
                nationality: requestInfo.nationality,
                isPcrPositive: requestInfo.is_pcr_positive,
                intensity: requestInfo.intensity,
                productivity: requestInfo.productivity
            });
            const data = Buffer.from(hl7, 'utf8');
            const filePath = `${config.hl7Folder}/${userId}/${requestId}/message.hl7`;
            fullPath = await fileService.saveFile(filePath, data);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.HL7_GENERATION_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
       
        res.status(HttpStatusCodes.SUCCESS);
        res.setHeader('Content-Type', 'text/*; charset=utf-8');
        try {
            const stream = await fileService
                .getFileAsStream(fullPath);
         
            stream.on('error', (err) => {
                const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            })
                .pipe(res);
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getReportByQR (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const qrToken = String(req.query.token);

        try {
            const reportResult = await fetchDiagnosticByQR(requestId, qrToken); 
            if (reportResult == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }
    
            const response: GetDiagnosticByIdResponse = {
                status: reportResult.status,
                date: reportResult.date,
                request_id: reportResult.request_id,
                age: reportResult.age,
                gender: reportResult.gender,
                identifier: reportResult.identifier,
                probability: reportResult.probability,
                diagnosis: reportResult.diagnosis
            };
    
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }

    public async getReportByQRV1_2 (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const qrToken = String(req.query.token);

        try {
            const reportResult = await fetchDiagnosticByQR(requestId, qrToken); 
            if (reportResult == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }
    
            const response: GetDiagnosticByIdResponse = {
                status: reportResult.status,
                date: reportResult.date,
                request_id: reportResult.request_id,
                age: reportResult.age,
                gender: reportResult.gender,
                identifier: reportResult.identifier,
                probability: reportResult.probability,
                diagnosis: reportResult.diagnosis,
                productivity: reportResult.productivity,
                intensity: reportResult.intensity
            };
    
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }
    
    public async deleteUserPdf (req: Request, res: Response) {
        const requestId = req.params.id;
    
        const diagnosticReportRepo = getRepository(DiagnosticReport);
        let report: DiagnosticReport;
        try {
            report = await diagnosticReportRepo.findOneOrFail({
                where: {
                    request_id: requestId,
                },
            });
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
        report.is_visible = false;
        await getManager().save(report);
        return res.status(HttpStatusCodes.SUCCESS).send();
    }
    
    public async updateCoughAudio (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = Number(req.token.userId);

        let requestBody: PatchCoughAudioRequest;
        try {
            requestBody = {
                cough_audio: req.files.cough_audio as UploadedFile
            };
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestRepo = getCustomRepository(DiagnosticRequestRepository);
        const statusRepo = getCustomRepository(DiagnosticRequestStatusRepository);
        const coughAudioRepo = getCustomRepository(DiagnosticCoughAudioRepository);

        let diagnosticRequest: DiagnosticRequest;
        try {
            diagnosticRequest = await requestRepo.findRequestByIdAndUserIdOrFail(requestId, userId);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }

        try {
            const status = await statusRepo.findByStatusIdOrFail(diagnosticRequest.status_id);
            if (status.request_status != DomainDiagnosticRequestStatus.NOISY_AUDIO) {
                const errorMessage = getErrorMessage(HttpErrors.NOT_NOISY);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            const coughAudio = await coughAudioRepo.findByRequestIdOrFail(requestId);
        
            const audioPath = `${config.audioFolder}/${userId}/${diagnosticRequest.id}/${requestBody.cough_audio.name}`;
            const fullPath = await fileService.saveFile(audioPath, requestBody.cough_audio.data);
            let isShouldDeletePrevAudio = false;
            let prevPath: string;
            if (coughAudio.file_path != fullPath) {
                prevPath = coughAudio.file_path;
                coughAudio.file_path = fullPath;
                isShouldDeletePrevAudio = true;
            }
            const pendingStatus = await statusRepo.findByStringOrFail(DomainDiagnosticRequestStatus.PROCESSING);
            diagnosticRequest.status_id = pendingStatus.id;
            await requestRepo.manager.save([diagnosticRequest, coughAudio]);
            if (isShouldDeletePrevAudio)
                await fileService.deleteFile(prevPath);
            
            inferenceDiagnostic(fullPath, requestId);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }
}
