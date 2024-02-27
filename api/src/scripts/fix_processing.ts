import {Connection, createConnection,  getRepository} from 'typeorm';
import {CoughAudio} from '../infrastructure/entity/CoughAudio';
import {DiagnosticRequest} from '../infrastructure/entity/DiagnosticRequest';
import {DiagnosticRequestStatusRepository} from '../infrastructure/repositories/diagnosticRequestStatusRepo';
import {getFileExtName} from '../helpers/file';
import {FileAccessService} from '../services/file/FileAccessService';
import config from '../config/config';
import {S3} from 'aws-sdk';
import {S3FileAccessServiceImpl} from '../services/file/S3FileAccessServiceImpl';
import axios from 'axios';
import FormData from 'form-data';
import {DiagnosticRequestStatus} from '../domain/RequestStatus';
import {DiagnosisTypes as EntityDiagnosisTypes} from '../infrastructure/entity/DiagnosisTypes';
import {DiagnosisTypes} from '../domain/DiagnosisTypes';
import {DiagnosticReport} from '../infrastructure/entity/DiagnosticReport';

let connection: Connection;
const PROCESSING_REQUESTS = [1788, 1791, 1798, 1852, 1856, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1876, 1891, 1902, 1905, 1907, 1914, 1917, 1920, 1924];

let fileService: FileAccessService;
const awsStandartParams = {
    accessKeyId: process.env.S3_USER_ID,
    secretAccessKey: process.env.S3_USER_SECRET,
    region: 'eu-central-1',
    endpoint: process.env.NODE_ENV == 'development' ? process.env.LOCALSTACK_ENDPOINT : undefined
};

if (config.fileAccessProtocol == 's3') {
    const s3 = new S3({
        ...awsStandartParams,
        s3ForcePathStyle: process.env.NODE_ENV == 'development' ? true : undefined,
    });
    fileService = new S3FileAccessServiceImpl(s3, config.s3Bucket, config.datasetSpectreFolder);
}
const inference = async (coughAudioPath: string, requestId: number) => {
    const diagnosticRequestRepo = connection.getRepository(DiagnosticRequest);
    const diagnosticRequest = await diagnosticRequestRepo.findOne({
        where: {id: requestId}
    });
    if (diagnosticRequest == undefined) {
        console.error('no such request');
        throw new Error('No such request');
    }
    const statusRepo = connection.getCustomRepository(DiagnosticRequestStatusRepository);

    try {
        const form = new FormData();
        const extension = getFileExtName(coughAudioPath);
        const stream = await fileService.getFileAsStream(coughAudioPath);
        form.append(
            'cough_audio',
            stream,
            {
                contentType: `audio/${extension}`,
                filename: `detection_audio.${extension}`,
            }
        );
        console.log('invoke ml');
        const response = await axios.post(`${config.mlServicePublicURL}/v1.1/public/inference/`, form, {
            headers: {...form.getHeaders(), timeout: config.mlServiceTimeout}
        });
        
        const coughAudioRepo = connection.getRepository(CoughAudio);
        const coughAudio = await coughAudioRepo.findOne({request_id: requestId});
        coughAudio.duration = response.data.audio_duration;
        coughAudio.samplerate = response.data.samplerate;
        const diagnosis = await connection.manager.findOneOrFail(EntityDiagnosisTypes, {
            where: {diagnosis_type: response.data.prediction > 0.5 ? DiagnosisTypes.COVID_19 : DiagnosisTypes.HEALTHY},
        });

        const pendingStatus = await statusRepo.findByStringOrFail(DiagnosticRequestStatus.PENDING);
        diagnosticRequest.status = pendingStatus;
        
        const reportRepo = connection.getRepository(DiagnosticReport);
        const diagnosticReport = await reportRepo.findOne({where: {request_id: requestId}});
        
        diagnosticReport.diagnosis_id = diagnosis.id;
        const probability = diagnosis.diagnosis_type == DiagnosisTypes.HEALTHY ?  1 - response.data.prediction 
            : response.data.prediction;
        diagnosticReport.diagnosis_probability = probability;

        await connection.manager.save([coughAudio, diagnosticRequest, diagnosticReport]);
        console.log('saved');
    } catch(error) {
        console.error(error.message);
    }
};

export const fix_processing = async () => {
    connection = await createConnection({
        type: 'postgres',
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        password: process.env.DB_PASSWORD,
        username: process.env.DB_USER,
        database: process.env.DB_NAME,
        logging: true,
        entities: ['dist/infrastructure/entity/*.js'],
        name: 'fix_processing'
    });
    try {
        for (const requestId of PROCESSING_REQUESTS) {
            console.log(`process ${requestId}`);
            const coughAudio = await connection.manager.findOne(CoughAudio, {
                where: {
                    request_id: requestId
                }
            });
            const audioPath = coughAudio.file_path;
            console.log('inference');
            await inference(audioPath, requestId);
        }
    } catch(error) {
        console.error(error);
    }

};

fix_processing();