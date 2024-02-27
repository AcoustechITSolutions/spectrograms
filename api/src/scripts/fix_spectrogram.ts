import {createConnection,  Not} from 'typeorm';

import {DatasetRequest} from '../infrastructure/entity/DatasetRequest';

import axios from 'axios';
import config from '../config/config';

export const fix_spectrogram = async () => {
    const connection = await createConnection({
        type: 'postgres',
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        password: process.env.DB_PASSWORD,
        username: process.env.DB_USER,
        database: process.env.DB_NAME,
        logging: true,
        entities: ['dist/infrastructure/entity/*.js'],
        name: 'spectr'
    });
    const records = await connection
        .manager
        .find(DatasetRequest, {
            relations: ['audio_info', 'audio_info.audio_type'],
            where: {user_id: Not(538)}
        });
    
    for (const record of records) {
        for (const record_audio of record.audio_info) {
            const audioPath = record_audio.audio_path;
            if (audioPath == undefined) 
                continue;
            console.log('preprocessing ', audioPath);
            const spectreFolder = `s3://${config.s3Bucket}/${config.datasetSpectreFolder}/${record.user_id}/${record.id}`;
            const reqBody = {
                audio_path: audioPath,
                audio_type: record_audio.audio_type.audio_type,
                spectre_folder: spectreFolder 
            };
            try {
                const resp = await axios.post(`${config.mlServiceURL}/v1/preprocess_audio/`, reqBody);
                record_audio.spectrogram_path = resp.data.spectrogram_path;
                console.log('path after preprocess ', record_audio.spectrogram_path);
                await connection.manager.save(record_audio);
            } catch(error) {
                console.error(error);
                continue;
            }
        }
    }    
};

fix_spectrogram();