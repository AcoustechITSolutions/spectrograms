import axios from 'axios';
import FormData from 'form-data';
import config from '../config/config';
import {Readable} from 'stream';

type Extensions = 'ogg' | 'wav'

interface NoisyCoughResponse {
    is_noisy: boolean
}

export const isNoisyCough = async (audioData: Buffer | Readable, extension: Extensions): Promise<boolean | undefined> => {
    try {
        const form = new FormData();
        form.append('cough_audio', audioData, 
            {contentType: `audio/${extension}`, filename: `noise_check.${extension}`});
        const response = await axios.post(`${config.mlServiceURL}/v1/noisy/cough/`, form, {
            headers: {...form.getHeaders()}
        });
        const body = response.data as NoisyCoughResponse;
        return body.is_noisy;
    } catch(error) {
        console.error(error.message);
        return Promise.resolve(undefined);
    }
};
