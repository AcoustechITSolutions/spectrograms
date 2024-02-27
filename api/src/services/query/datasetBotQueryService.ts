import {getConnection, getCustomRepository} from 'typeorm';
import {DatasetAudioInfo} from '../../infrastructure/entity/DatasetAudioInfo';
import {DatasetCoughCharacteristics} from '../../infrastructure/entity/DatasetCoughCharacteristics';
import {DatasetAudioTypes as DatasetAudioTypesEntity} from '../../infrastructure/entity/DatasetAudioTypes';
import {DatasetAudioTypes} from '../../domain/DatasetAudio';
import {CovidTypesRepository} from '../../infrastructure/repositories/covidTypesRepo';
import {DatasetPatientDetails} from '../../infrastructure/entity/DatasetPatientDetails';
import {DatasetPatientDiseases} from '../../infrastructure/entity/DatasetPatientDiseases';
import {DiseaseTypes} from '../../domain/DiseaseTypes';
import {DiseaseTypes as DiseaseTypesEntity} from '../../infrastructure/entity/DiseaseTypes';
import {TelegramDatasetRequest} from '../../infrastructure/entity/TelegramDatasetRequest';
import {DatasetRequest} from '../../infrastructure/entity/DatasetRequest';
import {DatasetRequestStatus as EntityDatasetRequestStatus} from '../../infrastructure/entity/DatasetRequestStatus';
import {DatasetRequestStatus as DomainDatasetRequestStatus} from '../../domain/RequestStatus';
import {DatasetMarkingStatus as EntityDatasetMarkingStatus} from '../../infrastructure/entity/DatasetMarkingStatus';
import {DatasetMarkingStatus as DomainDatasetMarkingStatus} from '../../domain/DatasetMarkingStatus';

export const persistAsDatasetRequest = async (req: TelegramDatasetRequest, userId: number) => { 
    const connection = getConnection(); 
    const preprocessingStatus = await connection.manager.findOneOrFail(EntityDatasetRequestStatus, {
        where: {request_status: DomainDatasetRequestStatus.PREPROCESSING},
    });
    const inProcessStatus = await connection.manager.findOneOrFail(EntityDatasetMarkingStatus, {
        where: {marking_status: DomainDatasetMarkingStatus.IN_PROCESS},
    });

    const healthyType = await getCustomRepository(CovidTypesRepository)
        .findByStringOrFail('no_covid19');
    const covidType = await getCustomRepository(CovidTypesRepository)
        .findByStringOrFail('covid19_mild_symptomatic');

    const acuteDisease = await connection.manager.findOneOrFail(DiseaseTypesEntity, {
        where: {disease_type: DiseaseTypes.ACUTE},
    });
    const noDisease = await connection.manager.findOneOrFail(DiseaseTypesEntity, {
        where: {disease_type: DiseaseTypes.NONE},
    });

    const audioType = await connection.manager.findOneOrFail(DatasetAudioTypesEntity, {
        where: {audio_type: DatasetAudioTypes.COUGH},
    });

    const datasetRequest = new DatasetRequest();
    datasetRequest.user_id = userId;
    datasetRequest.privacy_eula_version = 1;
    datasetRequest.status = preprocessingStatus;
    datasetRequest.is_visible = req.is_covid;
    datasetRequest.doctor_status = inProcessStatus;
    datasetRequest.marking_status = inProcessStatus;

    const patientDetails = new DatasetPatientDetails();
    patientDetails.age = req.age;
    patientDetails.gender_type_id = req.gender.id;
    patientDetails.is_smoking = req.is_smoking;
    patientDetails.identifier = '';
    patientDetails.request = datasetRequest;

    const patientDiseases = new DatasetPatientDiseases();
    if (req.is_covid) {
        patientDiseases.covid19_symptomatic_type_id = covidType.id;
    } else {
        patientDiseases.covid19_symptomatic_type_id = healthyType.id;
    }
    patientDiseases.other_disease_name = req.disease_name;
    if (req.is_disease) {
        patientDiseases.disease_type_id = acuteDisease.id;
    } else {
        patientDiseases.disease_type_id = noDisease.id;
    }
    patientDiseases.sick_days = 0;
    patientDiseases.request = datasetRequest;

    const coughCharacteristics = new DatasetCoughCharacteristics();
    coughCharacteristics.is_forced = req.is_forced;
    coughCharacteristics.request = datasetRequest;

    const coughAudio = new DatasetAudioInfo();
    coughAudio.audio_path = req.cough_audio_path;
    coughAudio.audio_type_id = audioType.id;
    coughAudio.request = datasetRequest;

    const queryRunner = connection.createQueryRunner();
    await queryRunner.startTransaction();

    const manager = queryRunner.manager;
    try {
        await manager.save([datasetRequest, coughCharacteristics, patientDiseases, patientDetails, coughAudio], {transaction: false});
        req.request = datasetRequest;
        await manager.save(req, {transaction: false});
        await queryRunner.commitTransaction();
        return datasetRequest.id;
    } catch(error) {
        console.error(error);
        await queryRunner.rollbackTransaction();
    } finally {
        await queryRunner.release();
    }
};
