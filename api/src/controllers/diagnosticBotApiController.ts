import {Request, Response} from 'express';
import {UploadedFile} from 'express-fileupload';
import {getErrorMessage, HttpErrors, HttpStatusCodes} from '../helpers/status';
import config from '../config/config';
import {TelegramDiagnosticRequest} from '../infrastructure/entity/TelegramDiagnosticRequest';
import {GenderTypes} from '../infrastructure/entity/GenderTypes';
import {getConnection, getCustomRepository} from 'typeorm';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';
import {TelegramDiagnosticRequestStatus} from '../infrastructure/entity/TelegramDiagnosticRequestStatus';
import {coughValidation} from '../services/coughValidation';
import {TelegramDiagnosticRequestStatusRepository} from '../infrastructure/repositories/telegramDiagnosticRequestStatusRepo';
import {DiagnosticRequestStatus, TelegramDiagnosticRequestStatus as DomainTelegramDiagnosticRequestStatus} from '../domain/RequestStatus';
import {isValidFileExt} from '../helpers/file';
import {persistAsDiagnosticRequest} from '../services/query/diagnosticBotQueryService';
import {inferenceDiagnostic} from '../services/diagnostic/diagnosticInferenceService';
import {fetchDiagnosticByTelegramChatId, fetchDiagnosticWithChatIdByRequestId, TelegramDiagnosticDatabaseReport} from '../services/query/diagnosticQueryService';
import {UserReportsResponse} from './diagnosticController';
import {fileService} from '../container';

const validExtensions = ['.ogg'];

type CreateDiagnosticRequest = {
    chat_id: number,
    cough_audio: UploadedFile,
    gender: string,
    is_smoking: boolean,
    is_forced: boolean,
    age: number
}

type DiagnosticByIdResponse = {
    status: string,
    commentary?: string,
    chat_id: number,
    diagnosis?: string,
    intensity?: string,
    productivity?: string,
    probability?: number
}

export const createDiagnostic = async (req: Request, res: Response) => {
    const userId = req.token.userId;
    let requestBody: CreateDiagnosticRequest;
    try {
        requestBody = {
            cough_audio: req.files.cough_audio as UploadedFile,
            chat_id: req.body.chat_id,
            gender: req.body.gender,
            is_smoking: JSON.parse(req.body.is_smoking),
            is_forced: JSON.parse(req.body.is_forced),
            age: req.body.age
        };
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    if (!isValidFileExt(validExtensions, requestBody?.cough_audio?.name)) {
        const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const validationResponse = await coughValidation(requestBody.cough_audio.data, 'ogg');
    if (validationResponse == undefined) {
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
    const isCough = validationResponse.is_cough;
    const isEnough = validationResponse.is_enough; 
    const isClear = validationResponse.is_clear;
    if (!isCough) {
        const errorMessage = getErrorMessage(HttpErrors.COUGH_NOT_DETECTED);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    if (!isEnough) {
        const errorMessage = getErrorMessage(HttpErrors.NOT_ENOUGH_COUGH);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    if (!isClear) {
        const errorMessage = getErrorMessage(HttpErrors.NOISY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    let gender: GenderTypes;
    try {
        gender = await getCustomRepository(GenderTypesRepository)
            .findByStringOrFail(requestBody.gender);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_GENDER);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    let status: TelegramDiagnosticRequestStatus;
    try {
        status = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
            .findByStringOrFail(DomainTelegramDiagnosticRequestStatus.DONE);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }

    const tgBotRequest = new TelegramDiagnosticRequest();
    tgBotRequest.gender = gender;
    tgBotRequest.is_forced = requestBody.is_forced;
    tgBotRequest.is_smoking = requestBody.is_smoking;
    tgBotRequest.age = requestBody.age;
    tgBotRequest.chat_id = requestBody.chat_id;
    tgBotRequest.status = status;
    
    const connection = getConnection();
    const queryRunner = connection.createQueryRunner();

    await queryRunner.startTransaction();
    try {
        const manager = queryRunner.manager;
        await manager.save(tgBotRequest, {transaction: false});

        const audio_path = `${config.tgBotAudioFolder}/${tgBotRequest.id}/cough.ogg`;
        const fullPath = await fileService.saveFile(audio_path, requestBody.cough_audio.data);
  
        tgBotRequest.cough_audio_path = fullPath;
        await manager.save(tgBotRequest, {transaction: false});
        await queryRunner.commitTransaction();
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    } finally {
        await queryRunner.release();
    }
    
    let diagnosticRequestId: number;
    try {
        diagnosticRequestId = await persistAsDiagnosticRequest(tgBotRequest, userId);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
    inferenceDiagnostic(tgBotRequest.cough_audio_path, diagnosticRequestId);
    return res.status(HttpStatusCodes.SUCCESS).send({
        'request_id': diagnosticRequestId
    });
};

export const getDiagnosticResultById = async (req: Request, res: Response) => {
    const userId = req.token.userId;
    const requestId = Number(req.params.id);
    let report: TelegramDiagnosticDatabaseReport;
    try {
        report = await fetchDiagnosticWithChatIdByRequestId(requestId, userId);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
    if (report == undefined) {
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    
    const response: DiagnosticByIdResponse = {
        status: report.status,
        chat_id: report.tg_old_chat_id
    };
    if (report.status == DiagnosticRequestStatus.SUCCESS) {
        response.commentary = report.commentary;
        response.diagnosis = report.diagnosis;
        response.intensity = report.intensity;
        response.probability = report.probability;
        response.productivity = report.productivity;
    }

    return res.status(HttpStatusCodes.SUCCESS).send(response);
};

export const getDiagnosticResults = async (req: Request, res: Response) => {
    const paginationParams = req.paginationParams;
    const userId = req.token.userId;

    const chatId = Number(req.query.chat_id);
    if (isNaN(chatId)) {
        const errorMessage = getErrorMessage(HttpErrors.NO_CHAT_ID);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    const userReports = await fetchDiagnosticByTelegramChatId(chatId, paginationParams, userId);
    const response: UserReportsResponse[] = userReports.map((userReport) => {
        const status = userReport.status;
        const date = userReport.date;
        const request_id = userReport.request_id;
        const report = status == DiagnosticRequestStatus.SUCCESS ? {
            probability: userReport.probability ?? undefined,
            diagnosis: userReport.diagnosis ?? undefined,
            productivity: userReport.productivity ?? undefined,
            intensity: userReport.intensity ?? undefined,
            commentary: userReport.commentary ?? undefined,
        } : undefined;
        return {status, date, request_id, report};
    });
    return res.status(HttpStatusCodes.SUCCESS).send(response);
};
