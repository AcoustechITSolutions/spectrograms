import axios from 'axios';
import FormData from 'form-data';
import config from '../config/config';
import {Readable} from 'stream';

type Extensions = 'ogg' | 'wav'

interface voiceEmbeddingResponse {
    voice_embedding: string
}

export const voiceEmbedding = async (audioData: Buffer | Readable, extension: Extensions): Promise<voiceEmbeddingResponse | undefined> => {
    try {
        const form = new FormData();
        form.append('speech_audio', audioData, 
            {contentType: `audio/${extension}`, filename: `voice.${extension}`});
        const response = await axios.post(`${config.mlServiceURL}/v1.2/voice/embedding/`, form, {
            headers: {...form.getHeaders()}
        });
        const body = response.data as voiceEmbeddingResponse;
        return body;
    } catch(error) {
        console.error(error.message);
        return Promise.resolve(undefined);
    }
};