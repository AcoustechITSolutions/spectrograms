import axios from 'axios';
import {DatasetAudioInfo} from '../infrastructure/entity/DatasetAudioInfo';
import {DatasetAudioTypes} from '../domain/DatasetAudio';
import {DatasetAudioInfoRepository} from '../infrastructure/repositories/datasetAudioInfoRepo';
import {DatasetRequest} from '../infrastructure/entity/DatasetRequest';
import {getConnection, getRepository, getCustomRepository} from 'typeorm';
import config from '../config/config';
import {DatasetRequestStatus as EntityDatasetRequestStatus} from '../infrastructure/entity/DatasetRequestStatus';
import {DatasetRequestStatus as DomainDatasetRequestStatus} from '../domain/RequestStatus';

interface PreprocessDataRequest {
    cough_audio_path: string,
    breathing_audio_path: string,
    speech_audio_path: string,
    spectre_folder: string
}

type PreprocessedAudioData = {
    samplerate: number,
    duration: number,
    spectre_path: string
}

interface PreprocessDataResponse {
    cough_audio: PreprocessedAudioData,
    breathing_audio: PreprocessedAudioData,
    speech_audio: PreprocessedAudioData
}

export const sendDataset = async (requestId: number) => {
    const datasetRequestRepo = getRepository(DatasetRequest);
    const datasetRequest = await datasetRequestRepo.findOne({
        where: {id: requestId}
    });
    if (datasetRequest == undefined) {
        console.error('no such request');
        throw new Error('No such request');
    }
    const connection = getConnection();

    const coughAudio = await getCustomRepository(DatasetAudioInfoRepository)
        .findAudioInfoRequestById(requestId, DatasetAudioTypes.COUGH);
    
    const breathingAudio =  await getCustomRepository(DatasetAudioInfoRepository)
        .findAudioInfoRequestById(requestId, DatasetAudioTypes.BREATHING);
    
    const speechAudio =  await getCustomRepository(DatasetAudioInfoRepository)
        .findAudioInfoRequestById(requestId, DatasetAudioTypes.SPEECH);

    const spectreFolder = `${config.datasetSpectreFolder}/${datasetRequest.user_id}/${datasetRequest.id}`;
    const reqBody: PreprocessDataRequest = {
        cough_audio_path: coughAudio?.audio_path,
        breathing_audio_path: breathingAudio?.audio_path,
        speech_audio_path: speechAudio?.audio_path,
        spectre_folder: spectreFolder
    };

    const preprocessingErrorStatus = await connection.manager.findOneOrFail(EntityDatasetRequestStatus, {
        where: {request_status: DomainDatasetRequestStatus.PREPROCESSING_ERROR},
    });

    try {
        const response = await axios.post(`${config.mlServiceURL}/v1/preprocess_datasetv2/`, reqBody, {
            timeout: config.mlServiceTimeout
        }); 
        const preprocessingResponse = response.data as PreprocessDataResponse;
        const updateEntities: any = [];
        if (coughAudio != undefined) {
            const updCoughAudio = new DatasetAudioInfo();
            updCoughAudio.id = coughAudio.id;
            updCoughAudio.audio_duration = preprocessingResponse.cough_audio.duration;
            updCoughAudio.samplerate = preprocessingResponse.cough_audio.samplerate;
            updCoughAudio.spectrogram_path = preprocessingResponse.cough_audio.spectre_path;
            updateEntities.push(updCoughAudio);
        }
        if (breathingAudio != undefined) {
            const updBreathingAudio = new DatasetAudioInfo();
            updBreathingAudio.id = breathingAudio.id;
            updBreathingAudio.audio_duration = preprocessingResponse.breathing_audio.duration;
            updBreathingAudio.samplerate = preprocessingResponse.breathing_audio.samplerate;
            updBreathingAudio.spectrogram_path = preprocessingResponse.breathing_audio.spectre_path;
            updateEntities.push(updBreathingAudio);
        }
        if (speechAudio != undefined) {
            const updSpeechAudio = new DatasetAudioInfo();
            updSpeechAudio.id = speechAudio.id;
            updSpeechAudio.audio_duration = preprocessingResponse.speech_audio.duration;
            updSpeechAudio.samplerate = preprocessingResponse.speech_audio.samplerate;
            updSpeechAudio.spectrogram_path = preprocessingResponse.speech_audio.spectre_path;
            updateEntities.push(updSpeechAudio);
        }

        const pendingStatus = await connection.manager.findOneOrFail(EntityDatasetRequestStatus, {
            where: {request_status: DomainDatasetRequestStatus.PENDING},
        });
        datasetRequest.status = pendingStatus;
        try {
            await connection.manager.save([...updateEntities, datasetRequest]);
        } catch (error) {
            console.error(error);
            datasetRequest.status = preprocessingErrorStatus;
            await connection.manager.save(datasetRequest, {transaction: false});
        }
    } catch (err) {
        console.error('dataset preprocessing error');
        console.error(err.message);
        datasetRequest.status = preprocessingErrorStatus;
        await connection.manager.save(datasetRequest, {transaction: false});
        return; 
    }
};
