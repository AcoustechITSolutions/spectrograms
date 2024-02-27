import axios from 'axios';
import moment from 'moment';
import {Request, Response} from 'express';
import {UploadedFile} from 'express-fileupload';
import FormData from 'form-data';
import {getConnection, getCustomRepository} from 'typeorm';
import config from '../config/config';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {getFileExtName, isValidFileExt} from '../helpers/file';
import {HWRequestStatus} from '../domain/RequestStatus';
import {HWCoughAudio} from '../infrastructure/entity/HWCoughAudio';
import {fileService} from '../container';
import {HWDiagnosticRequest as HWDiagnosticRequestEntity} from '../infrastructure/entity/HWDiagnosticRequest';
import {HWRequestStatus as HWRequestStatusEntity} from '../infrastructure/entity/HWRequestStatus';
import {HWDiagnosticReport} from '../infrastructure/entity/HWDiagnosticReport';
import {isCoughPresented} from '../services/coughDetector';
import {isNoisyCough} from '../services/noiseClassification';
import {coughValidation} from '../services/coughValidation';
import {voiceEmbedding} from '../services/voiceEmbedding';
import {UserRepository} from '../infrastructure/repositories/userRepo';
import {PersonalData} from '../infrastructure/entity/PersonalData';

const validExtensions = ['.wav'];

interface HWDiagnosticRequest {
    cough_audio: UploadedFile
}

interface VoiceAnalysisRequest {
    speech_audio: UploadedFile
}

interface HWDiagnosticResponse {
    diagnosis_probability: number,
    is_healthy: boolean
}

export const createHWDiagnostic = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: HWDiagnosticRequest = {
            cough_audio: req.files.cough_audio as UploadedFile,
        };
        const userId = req.token.userId;

        if (requestData.cough_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.cough_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const connection = getConnection();

        const processingStatus = await connection.manager.findOneOrFail(HWRequestStatusEntity, {
            where: {request_status: HWRequestStatus.PROCESSING},
        });
        const errorStatus = await connection.manager.findOneOrFail(HWRequestStatusEntity, {
            where: {request_status: HWRequestStatus.ERROR},
        });

        const hwRequest = new HWDiagnosticRequestEntity();
        hwRequest.user_id = userId;
        hwRequest.status = processingStatus;
        const report = new HWDiagnosticReport();
        report.request = hwRequest;
        await connection.manager.save([hwRequest, report]);
        const cough_audio_path = `${config.hwDiagnosticAudioFolder}/${userId}/${hwRequest.id}/${requestData.cough_audio.name}`;

        let fullPath: string;
        try {
            fullPath = await fileService.saveFile(cough_audio_path, requestData.cough_audio.data);
        } catch (error) {
            hwRequest.status = errorStatus;
            await connection.manager.save(hwRequest);
            const errorMessage = getErrorMessage(HttpErrors.FILE_SAVING_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }

        const coughAudio = new HWCoughAudio();
        coughAudio.file_path = fullPath;
        coughAudio.request = hwRequest;
        await connection.manager.save(coughAudio, {transaction: false});

        const form = new FormData();
        const extension = getFileExtName(requestData.cough_audio.name);
        form.append(
            'cough_audio',
            requestData.cough_audio.data,
            {
                contentType: `audio/${extension}`,
                filename: `detection_audio.${extension}`
            }
        );
 
        let response;
        try {
            response = await axios.post(`${config.mlServicePublicURL}/v1.1/public/inference/`, form, {
                headers: {...form.getHeaders()}
            });
        } catch(error) {
            console.error(`ml error, request id ${hwRequest.id}`);
            console.error(error.message);
            hwRequest.status = errorStatus;
            await connection.manager.save(hwRequest);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }

        coughAudio.duration = response.data.audio_duration;
        coughAudio.samplerate = response.data.samplerate;

        const successStatus = await connection.manager.findOneOrFail(HWRequestStatusEntity, {
            where: {request_status: HWRequestStatus.SUCCESS},
        });

        hwRequest.status = successStatus;
        report.diagnosis_probability = response.data.prediction;
        await connection.manager.save([hwRequest, report, coughAudio]);
        const hwResponse: HWDiagnosticResponse = {
            diagnosis_probability: report.diagnosis_probability,
            is_healthy: report.diagnosis_probability < 0.5,
        };
        
        return res.status(HttpStatusCodes.SUCCESS).send(hwResponse);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const createCoughDetectionCheck = async (req: Request, res: Response) => {
    try {
        let isValidate: boolean = true;
        if (req.token) {
            const userId = Number(req.token.userId);
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findOne(userId);
            isValidate = user.is_validate_cough;
        }
        
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: HWDiagnosticRequest = {
            cough_audio: req.files.cough_audio as UploadedFile,
        };
        if (requestData.cough_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.cough_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const detectionResponse = await isCoughPresented(requestData.cough_audio.data, 'wav');
        if (detectionResponse == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
        let isEnough: boolean = true;
        if (isValidate) {
            isEnough = detectionResponse.is_enough;
        }

        const isCough = detectionResponse.is_detected && isEnough;
 
        return res.status(HttpStatusCodes.SUCCESS).send({
            is_cough_detected: isCough,
            details: {
                is_cough: detectionResponse.is_detected,
                is_enough: isEnough
            }
        });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const createNoisyCoughCheck = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: HWDiagnosticRequest = {
            cough_audio: req.files.cough_audio as UploadedFile,
        };
        if (requestData.cough_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.cough_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const isNoisy = await isNoisyCough(requestData.cough_audio.data, 'wav');
        if (isNoisy == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
 
        return res.status(HttpStatusCodes.SUCCESS).send({
            is_noisy: isNoisy
        });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const createCoughValidationCheck = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: HWDiagnosticRequest = {
            cough_audio: req.files.cough_audio as UploadedFile,
        };
        if (requestData.cough_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.cough_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        let isValidate: boolean = true;
        if (req.token) {
            const userId = Number(req.token.userId);
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findOne(userId);
            isValidate = user.is_validate_cough;

            const currentDate = new Date();
            const formattedDate = moment(currentDate).format('YYYY-MM-DD:HH:mm:ss')
            const coughAudioPath = `${config.coughValidationFolder}/${userId}/${formattedDate}/${requestData.cough_audio.name}`;
            let fullPath: string;
            try {
                fullPath = await fileService.saveFile(coughAudioPath, requestData.cough_audio.data);
            } catch (error) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_SAVING_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }
        }

        const validationResponse = await coughValidation(requestData.cough_audio.data, 'wav');
        if (validationResponse == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
        let isEnough: boolean = true;
        if (isValidate) {
            isEnough = validationResponse.is_enough;
        }

        const isValid = validationResponse.is_cough && validationResponse.is_clear && isEnough;
 
        return res.status(HttpStatusCodes.SUCCESS).send({
            is_valid: isValid,
            details: {
                is_cough: validationResponse.is_cough,
                is_clear: validationResponse.is_clear,
                is_enough: isEnough
            }
        });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const createVoiceEmbedding = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: VoiceAnalysisRequest = {
            speech_audio: req.files.speech_audio as UploadedFile,
        };
        if (requestData.speech_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_SPEECH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.speech_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const embeddingResponse = await voiceEmbedding(requestData.speech_audio.data, 'wav');
        if (embeddingResponse == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NOT_ENOUGH_SPEECH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const newEmbedding = embeddingResponse.voice_embedding;

        if (req.token) {
            const userId = Number(req.token.userId);
            const connection = getConnection();
            const personalDataRepo = connection.getRepository(PersonalData);
            let userData: PersonalData;
            userData = await personalDataRepo.findOne({where: {user_id: userId}});
            if (userData == undefined) {
                userData = new PersonalData();
                userData.user_id = userId;
            }
            
            const audio_path = `${config.voiceFolder}/${userId}/${requestData.speech_audio.name}`;
            let fullPath: string;
            try {
                fullPath = await fileService.saveFile(audio_path, requestData.speech_audio.data);
            } catch (error) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_SAVING_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }
            userData.voice_embedding = newEmbedding;
            userData.voice_audio_path = fullPath;
            await personalDataRepo.save(userData);
        }

        return res.status(HttpStatusCodes.SUCCESS).send({
            voice_embedding: newEmbedding
        });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const createVoiceComparisonCheck = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: VoiceAnalysisRequest = {
            speech_audio: req.files.speech_audio as UploadedFile,
        };
        if (requestData.speech_audio == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_SPEECH_FILE);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        if (!isValidFileExt(validExtensions, requestData.speech_audio.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const userId = Number(req.token.userId);
        const connection = getConnection();
        const personalDataRepo = connection.getRepository(PersonalData);
        const userData = await personalDataRepo.findOne({where: {user_id: userId}});
        if (userData?.voice_embedding == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_PERSONAL_DATA);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
        const userEmbedding = userData.voice_embedding;
        const userEmbeddingArray: number[] = userEmbedding.split(',').map((element) => Number(element));

        const currentDate = new Date();
        const formattedDate = moment(currentDate).format('YYYY-MM-DD:HH:mm:ss')
        const speechAudioPath = `${config.voiceComparisonFolder}/${userId}/${formattedDate}/${requestData.speech_audio.name}`;
        let fullPath: string;
        try {
            fullPath = await fileService.saveFile(speechAudioPath, requestData.speech_audio.data);
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_SAVING_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }

        const embeddingResponse = await voiceEmbedding(requestData.speech_audio.data, 'wav');
        if (embeddingResponse == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NOT_ENOUGH_SPEECH);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const newEmbedding = embeddingResponse.voice_embedding;
        const newEmbeddingArray: number[] = newEmbedding.split(',').map((element) => Number(element));

        let similarity = 0;
        for (let i = 0; i < userEmbeddingArray.length; i++) {
            similarity += userEmbeddingArray[i] * newEmbeddingArray[i]
        }
        console.log(`Voice similarity: ${similarity}`);
        const isConfirmed = similarity > 0.75;
     
        return res.status(HttpStatusCodes.SUCCESS).send({
            is_confirmed: isConfirmed
        });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};
