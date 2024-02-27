import axios from 'axios';
import FormData from 'form-data';
import config from '../config/config';
import {Readable} from 'stream';

type Extensions = 'ogg' | 'wav'

interface CoughDetectionResponse {
    is_detected: boolean,
    is_enough: boolean
}

export const isCoughPresented = async (audioData: Buffer | Readable, extension: Extensions): Promise<CoughDetectionResponse | undefined> => {
    try {
        const form = new FormData();
        form.append('cough_audio', audioData, 
            {contentType: `audio/${extension}`, filename: `detection_audio.${extension}`});
        const response = await axios.post(`${config.mlServiceURL}/v1/detection/cough/`, form, {
            headers: {...form.getHeaders()}
        });
        return response.data as CoughDetectionResponse;
    } catch(error) {
        console.error(error.message);
        return Promise.resolve(undefined);
    }
};
