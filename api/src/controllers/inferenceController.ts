import axios from 'axios';
import {Request, Response} from 'express';
import {UploadedFile} from 'express-fileupload';
import FormData from 'form-data';
import config from '../config/config';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {getFileExtName, isValidFileExt} from '../helpers/file';

const validExtensions = ['.wav', '.ogg'];

interface InferenceRequest {
    cough_audio: UploadedFile
}

interface InferenceResponse {
    prediction: number,
    audio_duration: number,
    samplerate: number
}

export const inferenceController = async (req: Request, res: Response) => {
    try {
        if (req.files == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_FILES);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const requestData: InferenceRequest = {
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
            response = await axios.post(`${config.mlServiceURL}/v1.1/public/inference/`, form, {
                headers: {...form.getHeaders()}
            });
        } catch (error) {
            console.error(error.message);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }

        const inferenceResponse: InferenceResponse = {
            prediction: response.data.prediction,
            audio_duration: response.data.audio_duration,
            samplerate: response.data.samplerate,
        };

        return res.status(HttpStatusCodes.SUCCESS).send(inferenceResponse);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};
