import {Request, Response} from 'express';
import config from '../config/config';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';

import {UploadedFile} from 'express-fileupload';
import {isValidFileExt} from '../helpers/file';
import {getConnection, In} from 'typeorm';
import {DiseaseTypes} from '../domain/DiseaseTypes';

import {DiseaseTypes as DiseaseTypesEntity} from '../infrastructure/entity/DiseaseTypes';
import {AcuteCoughTypes as AcuteCoughTypesEntity} from '../infrastructure/entity/AcuteCoughTypes';
import {ChronicCoughTypes as ChronicCoughTypesEntity} from '../infrastructure/entity/ChronicCoughTypes';

import {DatasetRequest} from '../infrastructure/entity/DatasetRequest';
import {TelegramDatasetRequest} from '../infrastructure/entity/TelegramDatasetRequest';
import {DatasetPatientDetails} from '../infrastructure/entity/DatasetPatientDetails';
import {GenderTypes} from '../infrastructure/entity/GenderTypes';
import {DatasetPatientDiseases} from '../infrastructure/entity/DatasetPatientDiseases';
import {DatasetCoughCharacteristics} from '../infrastructure/entity/DatasetCoughCharacteristics';
import {DatasetAudioInfo} from '../infrastructure/entity/DatasetAudioInfo';
import {DatasetAudioTypes as DatasetAudioTypesEntity} from '../infrastructure/entity/DatasetAudioTypes';
import {DatasetAudioTypes} from '../domain/DatasetAudio';
import {DatasetRequestStatus as EntityDatasetRequestStatus} from '../infrastructure/entity/DatasetRequestStatus';
import {DatasetRequestStatus as DomainDatasetRequestStatus} from '../domain/RequestStatus';
import {DatasetMarkingStatus as EntityDatasetMarkingStatus} from '../infrastructure/entity/DatasetMarkingStatus';
import {DatasetMarkingStatus as DomainDatasetMarkingStatus} from '../domain/DatasetMarkingStatus';
import {DatasetAudioEpisodes} from '../infrastructure/entity/DatasetAudioEpisodes';
import {DatasetBreathingCharacteristics} from '../infrastructure/entity/DatasetBreathingCharacteristics';
import {DatasetSpeechCharacteristics} from '../infrastructure/entity/DatasetSpeechCharacteristics';
import {Covid19SymptomaticTypes as EntityCovid19SymptomaticTypes} from '../infrastructure/entity/Covid19SymptomaticTypes';

import {DatasetBreathingGeneralInfo} from '../infrastructure/entity/DatasetBreathingGeneralInfo';
import {DatasetAudioInfoRepository} from '../infrastructure/repositories/datasetAudioInfoRepo';
import {FileAccessService} from '../services/file/FileAccessService';
import {DatasetQueryService} from '../services/query/DatasetQueryService';
import {sendDataset} from '../services/sendDatasetService';

const validExtensions = ['.wav'];

interface CreateDatasetRequest {
    coughAudio: UploadedFile,
    breathAudio: UploadedFile,
    speechAudio: UploadedFile,
    age: number,
    gender: string,
    covid19_symptomatic_type?: string,
    isSmoking: boolean,
    isForce: boolean,
    privacyEulaVersion: number,
    sickDays: number,
    identifier: string,
    diseaseType: string,
    otherDiseaseName?: string,
    disease: string
}

interface CreateDatasetRequestV2 {
    coughAudio?: UploadedFile,
    breathAudio?: UploadedFile,
    speechAudio?: UploadedFile,
    age: number,
    gender: string,
    covid19_symptomatic_type?: string,
    isSmoking: boolean,
    isForce: boolean,
    privacyEulaVersion: number,
    sickDays: number,
    identifier: string,
    diseaseType: string,
    otherDiseaseName?: string,
    disease: string
}

interface CreateDatasetResponse {
    request_id: number
}

interface DeleteDatasetResponse {
    is_deleted: boolean
}

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

/**
 * Responsible for handling api calls to /dataset route. 
 * This route was built for user purposes, for admin processing see MarkingController
 */
export class DatasetController {
    //private mlService: MlService

    public constructor(private fileService: FileAccessService, private queryService: DatasetQueryService) {}

    public async createDatasetNullable (req: Request, res: Response) {
        try {
            const requestData: CreateDatasetRequestV2 = {
                coughAudio: req.files?.cough_audio as UploadedFile,
                breathAudio: req.files?.breath_audio as UploadedFile,
                speechAudio: req.files?.speech_audio as UploadedFile,
                age: req.body.age,
                gender: req.body.gender,
                covid19_symptomatic_type: req.body.covid19_symptomatic_type,
                isSmoking: JSON.parse(req.body.is_smoking),
                isForce: JSON.parse(req.body.is_force),
                privacyEulaVersion: req.body.privacy_eula_version,
                sickDays: req.body.sick_days,
                identifier: req.body.identifier,
                diseaseType: req.body.disease_type,
                otherDiseaseName: req.body.other_disease_name,
                disease: req.body.disease,
            };
        
            if ((requestData?.coughAudio != undefined && requestData?.coughAudio?.name == requestData?.breathAudio?.name) ||
                (requestData?.coughAudio != undefined && requestData?.coughAudio?.name == requestData?.speechAudio?.name) ||
                (requestData?.speechAudio != undefined && requestData?.speechAudio?.name == requestData?.breathAudio?.name)
            ) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_NAME_ERROR);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
    
            if ((requestData?.coughAudio?.name != undefined && !isValidFileExt(validExtensions, 
                requestData.coughAudio?.name)) 
            || (requestData?.breathAudio?.name != undefined && !isValidFileExt(validExtensions,
                requestData.breathAudio.name))
            || (requestData?.speechAudio?.name != undefined && !isValidFileExt(validExtensions,
                requestData.speechAudio?.name))
            ) {
                const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
    
            const connection = getConnection();
            let gender: GenderTypes;
            try {
                gender = await connection.manager.findOneOrFail(GenderTypes, {
                    where: {gender_type: requestData.gender},
                });
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_GENDER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
    
            let symptomaticType: EntityCovid19SymptomaticTypes = undefined;
            if (requestData.covid19_symptomatic_type != undefined) {
                try {
                    symptomaticType = await connection
                        .manager
                        .findOneOrFail(EntityCovid19SymptomaticTypes, {
                            where: {
                                symptomatic_type: requestData.covid19_symptomatic_type,
                            },
                        });
                } catch (error) {
                    console.error(error);
                    const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
            }
    
            let diseaseType: DiseaseTypesEntity;
            try {
                diseaseType = await connection.manager.findOneOrFail(DiseaseTypesEntity, {
                    where: {disease_type: requestData.diseaseType},
                });
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_DISEASE_TYPE);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            let acuteCoughType: AcuteCoughTypesEntity;
            let chronicCoughType: ChronicCoughTypesEntity;
    
            try {
                if (diseaseType.disease_type == DiseaseTypes.ACUTE) {
                    acuteCoughType = await connection.manager.findOneOrFail(AcuteCoughTypesEntity, {
                        where: {acute_cough_types: requestData.disease},
                    });
                } else if (diseaseType.disease_type == DiseaseTypes.CHRONIC) {
                    chronicCoughType = await connection.manager.findOneOrFail(ChronicCoughTypesEntity, {
                        where: {chronic_cough_type: requestData.disease},
                    });
                }
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_DISEASE_NAME);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
    
            const preprocessingStatus = await connection.manager.findOneOrFail(EntityDatasetRequestStatus, {
                where: {request_status: DomainDatasetRequestStatus.PREPROCESSING},
            });
            const breathingType = await connection.manager.findOneOrFail(DatasetAudioTypesEntity, {
                where: {audio_type: DatasetAudioTypes.BREATHING},
            });
            const coughType = await connection.manager.findOneOrFail(DatasetAudioTypesEntity, {
                where: {audio_type: DatasetAudioTypes.COUGH},
            });
            const speechType = await connection.manager.findOneOrFail(DatasetAudioTypesEntity, {
                where: {audio_type: DatasetAudioTypes.SPEECH},
            });
            const inProcessStatus = await connection.manager.findOneOrFail(EntityDatasetMarkingStatus, {
                where: {marking_status: DomainDatasetMarkingStatus.IN_PROCESS},
            });
    
            const userId = req.token.userId;
            const datasetRequest = new DatasetRequest();
    
            datasetRequest.user_id = userId;
            datasetRequest.privacy_eula_version = requestData.privacyEulaVersion;
            datasetRequest.status = preprocessingStatus;
            datasetRequest.is_visible = true;
            datasetRequest.doctor_status = inProcessStatus;
            datasetRequest.marking_status = inProcessStatus;
    
            const patientDetails = new DatasetPatientDetails();
            patientDetails.age = requestData.age;
            patientDetails.gender = gender;
            patientDetails.is_smoking = requestData.isSmoking;
            patientDetails.identifier = requestData.identifier;
            patientDetails.request = datasetRequest;
    
            const patientDiseases = new DatasetPatientDiseases();
            patientDiseases.covid19_symptomatic_type = symptomaticType;
            patientDiseases.other_disease_name = requestData.otherDiseaseName;
            patientDiseases.disease_type = diseaseType;
            patientDiseases.acute_cough_types = acuteCoughType;
            patientDiseases.chronic_cough_types = chronicCoughType;
            patientDiseases.sick_days = requestData.sickDays;
            patientDiseases.request = datasetRequest;
    
            const coughCharacteristics = new DatasetCoughCharacteristics();
            coughCharacteristics.is_forced = requestData.isForce;
            coughCharacteristics.request = datasetRequest;
    
            let coughAudio: DatasetAudioInfo;
            let breathingAudio: DatasetAudioInfo;
            let speechAudio: DatasetAudioInfo;
    
            try {
                await connection.manager.save([datasetRequest, coughCharacteristics, patientDiseases, patientDetails]);
                const audioEntities = [];
                if (requestData?.coughAudio != undefined) {
                    const coughAudioPath = `${config.dataset_audio_folder}/${userId}/${datasetRequest.id}/${requestData.coughAudio.name}`;
                    const fullPath = await this.fileService.saveFile(coughAudioPath, requestData.coughAudio.data);
                    coughAudio = new DatasetAudioInfo();
                    coughAudio.audio_type = coughType;
                    coughAudio.audio_path = fullPath;
                    coughAudio.request = datasetRequest;
                    audioEntities.push(coughAudio);
                }
    
                if (requestData?.breathAudio != undefined) {
                    const breathAudioPath = `${config.dataset_audio_folder}/${userId}/${datasetRequest.id}/${requestData.breathAudio.name}`;
                    const fullPath = await this.fileService.saveFile(breathAudioPath, requestData.breathAudio.data);
                    breathingAudio = new DatasetAudioInfo();
                    breathingAudio.audio_type = breathingType;
                    breathingAudio.audio_path = fullPath;
                    breathingAudio.request = datasetRequest;
                    audioEntities.push(breathingAudio);
                }
    
                if (requestData?.speechAudio != undefined) {
                    const speechAudioPath = `${config.dataset_audio_folder}/${userId}/${datasetRequest.id}/${requestData.speechAudio.name}`;
                    const fullPath = await this.fileService.saveFile(speechAudioPath, requestData.speechAudio.data);
                    speechAudio = new DatasetAudioInfo();
                    speechAudio.audio_type = speechType;
                    speechAudio.audio_path = fullPath;
                    speechAudio.request = datasetRequest;
                    audioEntities.push(speechAudio);
                }
                           
                await connection.manager.save(audioEntities);
                const response: CreateDatasetResponse = {
                    request_id: datasetRequest.id,
                };
                sendDataset(datasetRequest.id);
                return res.status(HttpStatusCodes.CREATED).send(response);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }

        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }
    
    public async getDatasetRecords (req: Request, res: Response) {
        try {
            const userId = req.token.userId;
            const paginationParams = req.paginationParams;
            
            const userDatasetRecords = await this.queryService
                .fetchDatasetRecordByUserId(userId, paginationParams);
               
            return res.status(HttpStatusCodes.SUCCESS).send(userDatasetRecords);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }
    
    public async deleteUserDataset (req: Request, res: Response) {
        try {
            const userId = req.token.userId;
            const requestId = Number(req.params.id);
    
            const connection = getConnection();
            const queryRunner = connection.createQueryRunner();
    
            await queryRunner.startTransaction();
            const manager = queryRunner.manager;
    
            try {
                const coughAudioReq = manager.findOneOrFail(DatasetAudioInfo, {select: ['id', 'audio_path', 'spectrogram_path'], where: {request_id: requestId}});
                const datasetRequestReq = manager.findOneOrFail(DatasetRequest, {select: ['user_id', 'id'], where: {id: requestId}});
                const audioIdsReq = manager.getCustomRepository(DatasetAudioInfoRepository).findAudioIdsByRequestId(requestId);
    
                const dbResult = await Promise.all([coughAudioReq, datasetRequestReq, audioIdsReq]);
                const coughAudio = dbResult[0];
                const datasetRequest = dbResult[1];
                const audioIds = dbResult[2];
    
                if (datasetRequest.user_id != userId) {
                    await queryRunner.rollbackTransaction();
                    const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN_TO_DELETE);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                
                await manager.delete(DatasetBreathingGeneralInfo, {request_id: requestId});
                await manager.delete(DatasetSpeechCharacteristics, {request_id: requestId});
                await manager.delete(DatasetBreathingCharacteristics, {request_id: requestId});
                await manager.delete(DatasetCoughCharacteristics, {request_id: requestId});
                await manager.delete(DatasetPatientDetails, {request_id: requestId});
                await manager.delete(DatasetPatientDiseases, {request_id: requestId});
                await manager.delete(DatasetAudioEpisodes, {audio_info_id: In(audioIds)});
                await manager.delete(DatasetAudioInfo, {request_id: requestId});
                await manager.delete(TelegramDatasetRequest, {request_id: requestId});
                await manager.delete(DatasetRequest, {id: requestId});
                
                if (coughAudio?.spectrogram_path != undefined) {
                    await this.fileService.deleteDirectory(coughAudio.spectrogram_path);
                }
                if (coughAudio?.audio_path != undefined) {
                    await this.fileService.deleteDirectory(coughAudio.audio_path);
                }
                await queryRunner.commitTransaction();
    
                const response: DeleteDatasetResponse = {
                    is_deleted: true,
                };
                return res.status(HttpStatusCodes.SUCCESS).send(response);
            } catch (error) {
                console.error(error);
                await queryRunner.rollbackTransaction();
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            } finally {
                await queryRunner.release();
            }
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }
}

