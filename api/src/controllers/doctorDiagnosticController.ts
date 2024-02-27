import {Request, Response} from 'express';
import config from '../config/config';

import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {DiagnosticRequest} from '../infrastructure/entity/DiagnosticRequest';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus} from '../domain/RequestStatus';
import {getRepository, getConnection,  getCustomRepository} from 'typeorm';
import {CoughAudio} from '../infrastructure/entity/CoughAudio';
import {PatientInfo} from '../infrastructure/entity/PatientInfo';
import {getFileExtName, isValidFileExt} from '../helpers/file';
import {UploadedFile} from 'express-fileupload';
import {DiagnosticReport} from '../infrastructure/entity/DiagnosticReport';
import {CoughCharacteristics} from '../infrastructure/entity/CoughCharacteristic';

import {fileService} from '../container';
import {DiagnosticRequestStatus as EntityDiagnosticRequestStatus} from '../infrastructure/entity/DiagnostRequestStatus';
import {DiagnosticRequestStatusRepository} from '../infrastructure/repositories/diagnosticRequestStatusRepo';

import {inferenceDiagnostic} from '../services/diagnostic/diagnosticInferenceService';

import {fetchCoughAudioPath, fetchDiagnostic, fetchDiagnosticById} from '../services/query/diagnosticQueryService';

import {UserSubscriptionsRepository} from '../infrastructure/repositories/userSubscriptionsRepo';
import {Gender} from '../domain/Gender';
import {DiagnosticRequestRepository} from '../infrastructure/repositories/diagnosticRequestRepo';
import {DiagnosticCoughAudioRepository} from '../infrastructure/repositories/diagnosticCoughAudioRepo';

export type UserReportsResponse = {
    status: string,
    date: Date,
    request_id: number,
    report?: UserDiagnosticReport
}

export type UserDiagnosticReport = {
    probability?: number,
    diagnosis?: string,
    productivity?: string,
    intensity?: string,
    commentary?: string
}

type PatchCoughAudioRequest = {
    cough_audio: UploadedFile
}

const validExtensions = ['.wav', '.ogg'];

export class DoctorDiagnosticController {

    public async createDiagnostic(
        req: Request, 
        res: Response
    ) {
        try {
            let audio: UploadedFile;
            let language: string;
            try {
                audio = req.files.cough_audio as UploadedFile;
                language = req.body.language;
            } catch(error) {
                const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            const userId = req.token.userId;

            if (!isValidFileExt(validExtensions, audio.name)) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }

            if (language == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_LANGUAGE);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            const subscriptionRepo = getCustomRepository(UserSubscriptionsRepository);
            const subscription = await subscriptionRepo.findByUserId(userId);
            if (subscription == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_SUBSCRIPTION);
                return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
            }

            // Sanity check
            if (subscription.diagnostics_left <= 0) {
                const errorMessage = getErrorMessage(HttpErrors.NO_DIAGNOSTICS);
                return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
            }
            
            const statusRepo = getCustomRepository(DiagnosticRequestStatusRepository);
            let processingStatus: EntityDiagnosticRequestStatus;
            try {
                processingStatus = await statusRepo.findByStringOrFail(DomainDiagnosticRequestStatus.PROCESSING);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }

            const diagnosticRequest = new DiagnosticRequest();
            diagnosticRequest.user_id = userId;
            diagnosticRequest.language = req.language; 
            diagnosticRequest.status_id = processingStatus.id;

            // TODO: remove personal data from processing and after remove this
            const patientInfo = new PatientInfo();
            patientInfo.age = 0;
            patientInfo.gender = Gender.MALE;
            patientInfo.is_smoking = false;
            patientInfo.request = diagnosticRequest;
            patientInfo.sick_days = 0;
    
            const coughCharacteristic = new CoughCharacteristics();
            coughCharacteristic.request = diagnosticRequest;
            coughCharacteristic.is_forced = false;
            const queryRunner = getConnection().createQueryRunner();
            await queryRunner.startTransaction();
            const entityManager = queryRunner.manager;
            let full_path: string = undefined;
            try {
                await entityManager.save([diagnosticRequest, patientInfo, coughCharacteristic]);
                const coughAudio = new CoughAudio();
                const audio_path = `${config.audioFolder}/${userId}/${diagnosticRequest.id}/${audio.name}`;
                coughAudio.request = diagnosticRequest;
                coughAudio.file_path = await fileService.saveFile(audio_path, audio.data);
                full_path = coughAudio.file_path;

                const diagnosticReport = new DiagnosticReport();
                diagnosticReport.user_id = userId;
                diagnosticReport.request = diagnosticRequest;
                await entityManager.save([coughAudio, diagnosticReport]);
            
                const subsLocked = await subscriptionRepo.findByIdLocked(subscription.id, entityManager);
                if (subsLocked.diagnostics_left <= 0) {
                    console.log(audio_path);
                    console.log(coughAudio.file_path);
                    await entityManager.queryRunner.rollbackTransaction();
                    fileService.deleteDirectory(full_path);
                    const errorMessage = getErrorMessage(HttpErrors.NO_DIAGNOSTICS);
                    return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
                }
                subsLocked.diagnostics_left = subsLocked.diagnostics_left - 1;
                await entityManager.save(subsLocked);
                await queryRunner.commitTransaction();
                // TODO: replace with rmq enqueue
                inferenceDiagnostic(coughAudio.file_path, diagnosticRequest.id);
            
                return res.status(HttpStatusCodes.CREATED).send({'request_id': diagnosticRequest.id});
            } catch(error) {
                console.error(error);
                await queryRunner.rollbackTransaction();
                if (full_path != undefined) {
                    await fileService.deleteDirectory(full_path);
                }
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            } finally {
                await queryRunner.release();
            }
        } catch(error) {
            console.error(error);
            return res.status(HttpStatusCodes.ERROR).send();
        }
    }

    public async getDiagnostic(req: Request, res: Response) {
        const userId = req.token.userId;
        const byIdParam = req.query.by_id;
        let byIds = undefined;
        if (typeof byIdParam == 'string') {
            byIds = [Number.parseInt(byIdParam)];
        } else if (Array.isArray(byIdParam) && typeof byIdParam[0] == 'string') {
            byIds = (byIdParam as string[]).map(x => Number.parseInt(x));
        }
        const paginationParams = req.paginationParams;
    
        try {
            const userReportsResult = await fetchDiagnostic(userId, paginationParams, byIds);     
    
            const response: UserReportsResponse[] = userReportsResult?.map((userReport) => {
                const status = userReport.status;
                const date = userReport.date;
                const request_id = userReport.request_id;
                const report = status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    probability: userReport.probability ?? undefined,
                    diagnosis: userReport.diagnosis ?? undefined,
                    productivity: userReport.productivity ?? undefined,
                    intensity: userReport.intensity ?? undefined,
                    commentary: userReport.commentary ?? undefined,
                } : undefined;
                return {status, date, request_id, report};
            });
    
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }

    public async getDiagnosticById(req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = req.token.userId;

        try {
            const dbResponse = await fetchDiagnosticById(userId, requestId);
            if (dbResponse == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }

            const response: UserReportsResponse = {
                status: dbResponse.status,
                date: dbResponse.date,
                request_id: dbResponse.request_id,
                report: dbResponse.status == DomainDiagnosticRequestStatus.SUCCESS ? {
                    intensity: dbResponse.intensity,
                    diagnosis: dbResponse.diagnosis,
                    probability: dbResponse.probability,
                    commentary: dbResponse.commentary,
                    productivity: dbResponse.productivity
                } : undefined
            };

            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getDiagnosticCoughAudio (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const userId = req.token.userId;

        const coughAudioRepo = getRepository(CoughAudio);
        let filePath: string;
        try {
            filePath = await fetchCoughAudioPath(userId, requestId);
        } catch (error) {
            console.error(error);
            res.setHeader('Content-Type', 'application/json');
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        res.status(HttpStatusCodes.SUCCESS);
        const extension = getFileExtName(filePath);
        res.setHeader('Content-Type', `audio/${extension}`);
        try {
            const stream = await fileService
                .getFileAsStream(filePath);
    
            stream
                .on('error', (err) => {
                    console.error(err);
                    const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                    res.setHeader('Content-Type', 'application/json');
                    return res.status(HttpStatusCodes.ERROR).send(errorMessage);
                })
                .pipe(res);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
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
