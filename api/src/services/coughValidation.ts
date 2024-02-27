import axios from 'axios';
import FormData from 'form-data';
import config from '../config/config';
import {Readable} from 'stream';

type Extensions = 'ogg' | 'wav'

interface coughValidationResponse {
    is_cough: boolean
    is_clear: boolean,
    is_enough: boolean
}

export const coughValidation = async (audioData: Buffer | Readable, extension: Extensions): Promise<coughValidationResponse | undefined> => {
    try {
        const form = new FormData();
        form.append('cough_audio', audioData, 
            {contentType: `audio/${extension}`, filename: `validation.${extension}`});
        const response = await axios.post(`${config.mlServiceURL}/v1/validation/cough/`, form, {
            headers: {...form.getHeaders()}
        });
        const body = response.data as coughValidationResponse;
        return body;
    } catch(error) {
        console.error(error.message);
        return Promise.resolve(undefined);
    }
};