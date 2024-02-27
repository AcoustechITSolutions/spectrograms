import axios from 'axios';
import FormData from 'form-data';
import moment from 'moment';
import {CoughAudio} from '../../infrastructure/entity/CoughAudio';
import {DiagnosticRequest} from '../../infrastructure/entity/DiagnosticRequest';
import {getConnection, getRepository, getCustomRepository} from 'typeorm';
import config from '../../config/config';
import {DiagnosticRequestStatusRepository} from '../../infrastructure/repositories/diagnosticRequestStatusRepo';
import {DiagnosisTypes as EntityDiagnosisTypes} from '../../infrastructure/entity/DiagnosisTypes';
import {DiagnosisTypes} from '../../domain/DiagnosisTypes';
import {DiagnosticReport} from '../../infrastructure/entity/DiagnosticReport';
import {doctorNotificationService} from '../../container';
import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {onDiagnosticError, onPatchDiagnostic} from './diagnosticBotNotificationService';
import {onNewDiagnosticError, onPatchNewDiagnostic} from './newDiagnosticBotNotificationService';
import {onMuusDiagnosticError, onPatchMuusDiagnostic} from './muusBotNotificationService';
import {fileService} from '../../container';
import {UserRepository} from '../../infrastructure/repositories/userRepo';

import {getFileExtName, getFileName} from '../../helpers/file';

export const inferenceDiagnostic = async (coughAudioPath: string, requestId: number) => {
    const diagnosticRequestRepo = getRepository(DiagnosticRequest);
    const diagnosticRequest = await diagnosticRequestRepo.findOne({
        where: {id: requestId}
    });
    if (diagnosticRequest == undefined) {
        console.error('no such request');
        throw new Error('No such request');
    }
    const connection = getConnection();
    const statusRepo = connection.getCustomRepository(DiagnosticRequestStatusRepository);
    const userId = diagnosticRequest.user_id;

    try {
        const basename = getFileName(coughAudioPath);
        const extension = getFileExtName(coughAudioPath);
        const stream = await fileService.getFileAsStream(coughAudioPath);
        const convert_form = new FormData();
        convert_form.append(
            'file',
            stream,
            {
                contentType: `audio/${extension}`,
                filename: `converter_file.${extension}`
            }
        );
        const converted = await axios.post(`${config.mlServiceURL}/v1.1/convert/`, convert_form, {
            headers: {...convert_form.getHeaders()},
            responseType: 'stream'
        });

        const newAudioPath = `${config.audioFolder}/${userId}/${diagnosticRequest.id}/${basename}_converted.wav`;
        const chunks = [];
        for await (const chunk of converted.data) {
            chunks.push(chunk);
        }
        const buffer = Buffer.concat(chunks);
        const coughAudioRepo = getRepository(CoughAudio);
        const coughAudio = await coughAudioRepo.findOne({request_id: requestId});
        coughAudio.file_path = await fileService.saveFile(newAudioPath, buffer);
        
        const form = new FormData();
        form.append(
            'cough_audio',
            buffer,
            {
                contentType: `audio/wav`,
                filename: `detection_audio.wav`,
            }
        );
        const response = await axios.post(`${config.mlServicePublicURL}/v1.1/public/inference/`, form, {
            headers: {...form.getHeaders(), timeout: config.mlServiceTimeout}
        });
        const spectreFolder = `${config.spectreFolder}/${diagnosticRequest.user_id}/${diagnosticRequest.id}`;
        const reqBody = {
            cough_audio_path: coughAudioPath,
            spectre_folder: spectreFolder
        };
        const spectreResponse = await axios.post(`${config.mlServiceURL}/v1.2/spectrogram/`, reqBody, {
            timeout: config.mlServiceTimeout
        });
        
        coughAudio.duration = response.data.audio_duration;
        coughAudio.samplerate = response.data.samplerate;
        coughAudio.spectrogram_path = spectreResponse.data.spectre_path;
        const connection = getConnection();
        const diagnosis = await connection.manager.findOneOrFail(EntityDiagnosisTypes, {
            where: {diagnosis_type: response.data.prediction > 0.5 ? DiagnosisTypes.COVID_19 : DiagnosisTypes.HEALTHY},
        });

        const reportRepo = getRepository(DiagnosticReport);
        const diagnosticReport = await reportRepo.findOne({where: {request_id: requestId}});
        
        diagnosticReport.diagnosis_id = diagnosis.id;
        const probability = diagnosis.diagnosis_type == DiagnosisTypes.HEALTHY ?  1 - response.data.prediction 
            : response.data.prediction;
        diagnosticReport.diagnosis_probability = probability;

        await connection.manager.save([coughAudio, diagnosticReport]);

        const user = await getCustomRepository(UserRepository).findOne(userId);
        let isDoctorChecking = true;
        if (user.check_start && user.check_end) {
            const currentDate = new Date();
            const formattedDate = moment(currentDate).format('YYYY-MM-DD');
            const todayCheckStart = moment(`${formattedDate} ${user.check_start}`, 'YYYY-MM-DD HH:mm');
            const todayCheckEnd = moment(`${formattedDate} ${user.check_end}`, 'YYYY-MM-DD HH:mm');
            isDoctorChecking = moment(currentDate).isBetween(todayCheckStart, todayCheckEnd);
        }
        if ((diagnosis.diagnosis_type == DiagnosisTypes.HEALTHY && !user.is_check_healthy) ||
            (diagnosis.diagnosis_type == DiagnosisTypes.COVID_19 && !user.is_check_covid) || 
            !isDoctorChecking) {
            diagnosticRequest.status = await statusRepo.findByStringOrFail(DiagnosticRequestStatus.SUCCESS);
            await connection.manager.save(diagnosticRequest);
            onPatchDiagnostic(Number(requestId));
            onPatchNewDiagnostic(Number(requestId));
            onPatchMuusDiagnostic(Number(requestId));
        } else {
            diagnosticRequest.status = await statusRepo.findByStringOrFail(DiagnosticRequestStatus.PENDING);
            await connection.manager.save(diagnosticRequest);
            await doctorNotificationService.notifyAboutNewDiagnostic();
        }
    } catch(error) {
        console.error(error.message);
        const errorStatus = await statusRepo.findByStringOrFail(DiagnosticRequestStatus.ERROR);
        diagnosticRequest.status_id = errorStatus.id;
        await connection.manager.save(diagnosticRequest, {transaction: false});
        onDiagnosticError(diagnosticRequest.id);
        onNewDiagnosticError(diagnosticRequest.id);
        onMuusDiagnosticError(diagnosticRequest.id);
    }
};
