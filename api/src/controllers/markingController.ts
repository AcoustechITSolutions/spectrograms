import {Request, Response} from 'express';

import {getConnection, getCustomRepository, EntityManager, In} from 'typeorm';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {DatasetRequest} from '../infrastructure/entity/DatasetRequest';
import {TelegramDatasetRequest} from '../infrastructure/entity/TelegramDatasetRequest';
import {DatasetAudioInfo} from '../infrastructure/entity/DatasetAudioInfo';
import {DatasetAudioTypes} from '../domain/DatasetAudio';
import {DatasetMarkingStatus} from '../domain/DatasetMarkingStatus';
import {DiseaseTypes, AcuteCoughTypes,  ChronicCoughTypes,  getDiagnosisName} from '../domain/DiseaseTypes';
import {getTypeFromString} from '../domain/DatasetAudio';
import {DatasetPatientDetails} from '../infrastructure/entity/DatasetPatientDetails';
import {DatasetPatientDiseases} from '../infrastructure/entity/DatasetPatientDiseases';
import {Gender} from '../domain/Gender';

import {BreathingDepth} from '../domain/BreathingCharacteristics';
import {BreathingTypes} from '../domain/BreathingCharacteristics';
import {BreathingDuration} from '../domain/BreathingCharacteristics';
import {BreathingDifficulty} from '../domain/BreathingCharacteristics';
import {fileService} from '../container';
import {Covid19SymptomaticTypes} from '../domain/Covid19Types';
import {Covid19SymptomaticTypes as Covid19Types} from '../infrastructure/entity/Covid19SymptomaticTypes';
import {GenderTypes} from '../infrastructure/entity/GenderTypes';
import {DatasetMarkingStatus as EntityDatasetMarkingStatus} from '../infrastructure/entity/DatasetMarkingStatus';
import {NoiseTypes} from '../infrastructure/entity/NoiseTypes';
import {CoughIntensityTypes} from '../infrastructure/entity/CoughIntensityTypes';
import {CoughProductivityTypes} from '../infrastructure/entity/CoughProductivityTypes';
import {DatasetCoughCharacteristics} from '../infrastructure/entity/DatasetCoughCharacteristics';
import {DatasetSpeechCharacteristics} from '../infrastructure/entity/DatasetSpeechCharacteristics';
import {DatasetBreathingCharacteristics} from '../infrastructure/entity/DatasetBreathingCharacteristics';
import {BreathingDepthTypes as EntityBreathingDepthTypes} from '../infrastructure/entity/BreathingDepthTypes';
import {BreathingDifficultyTypes as EntityBreathingDifficultyTypes} from '../infrastructure/entity/BreathingDifficultyTypes';
import {BreathingDurationTypes as EntityBreathingDurationTypes} from '../infrastructure/entity/BreathingDurationTypes';
import {BreathingDepthTypesRepository} from '../infrastructure/repositories/breathingDepthTypeRepo';
import {BreathingDifficultyTypesRepository} from '../infrastructure/repositories/breathingDifficultyTypeRepo';
import {BreahtingDurationTypesRepository} from '../infrastructure/repositories/breathingDurationTypesRepo';
import {BreathingTypesRepository} from '../infrastructure/repositories/breathingTypesRepo';
import {DatasetAudioEpisodesRepository} from '../infrastructure/repositories/datasetAudioEpisodesRepo';
import {DatasetAudioEpisodes} from '../infrastructure/entity/DatasetAudioEpisodes';
import {DatasetAudioInfoRepository} from '../infrastructure/repositories/datasetAudioInfoRepo';
import {createSortByRegExp, getSortingParamsByRegExp} from '../helpers/sort';
import {fetchAudioParams, fetchBreathingDetailed, fetchCoughDetailed, fetchMarkingGeneral, fetchNavigationByRequestId, fetchSpeechDetailed} from '../services/query/markingQueryService';
import {DatasetBreathingGeneralInfo} from '../infrastructure/entity/DatasetBreathingGeneralInfo';
import {DatasetEpisodesTypes as DomainDatasetEpisodesTypes} from '../domain/DatasetEpisodesTypes';
import {DatasetEpisodesTypes} from '../infrastructure/entity/DatasetEpisodesTypes';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';
import {CovidTypesRepository} from '../infrastructure/repositories/covidTypesRepo';
import {DatasetPatientDiseasesRepository} from '../infrastructure/repositories/datasetPatientDiseasesRepo';
import {UserRepository} from '../infrastructure/repositories/userRepo';

import {getFileExtName} from '../helpers/file';

type MarkingResponse = {
    date_created: string,
    doctor_status: string,
    marking_status: string,
    request_id: number,
    data_source: string,
    full_diagnosis: string,
    covid_status: string,
    identifier: string,
    is_marked_doctor_cough: boolean,
    is_marked_doctor_breathing: boolean,
    is_marked_doctor_speech: boolean,
    is_marked_scientist_cough: boolean,
    is_marked_scientist_breathing: boolean,
    is_marked_scientist_speech: boolean
} 

type AudioGeneralInfoResponse = {
    age: number,
    gender: string,
    is_smoking: boolean,
    marking_status: string,
    doctor_status: string,
    data_source: string,
    full_diagnosis: string,
    covid19_symptomatic_type?: string,
    request_id: number
}

type AudioGeneralInfoDatabaseResponse = {
    file_path: string,
    marking_status: DatasetMarkingStatus,
    doctor_status: DatasetMarkingStatus,
    data_source: string,
    disease_type: DiseaseTypes,
    acute_cough_type?: AcuteCoughTypes,
    chronic_cough_type?: ChronicCoughTypes,
    other_disease_name?: string,
    covid19_symptomatic_type?: Covid19SymptomaticTypes,
    age: number,
    gender: Gender,
    is_smoking: boolean,
    request_id: number
}

type AudioEpisode = {
    start: number,
    end: number,
    id: number,
    type: string
}

type AudioParams = {
    is_representative?: boolean,
    is_representative_scientist?: boolean,
    is_validation_audio?: boolean,
    samplerate: number,
    audio_path: string,
    spectrogram_path: string,
    is_marked: boolean,
    is_marked_scientist: boolean,
    episodes: AudioEpisode[],
    noise_type: string
}

type CoughDetailedResponse = {
    audio_params: AudioParams,
    productivity: string,
    intensity: string,
    symptom_duration: number,
    commentary: string
}

type BreathingCharactersticsResponse = {
    depth_type: BreathingDepth,
    difficulty_type: BreathingDifficulty,
    duration_type: BreathingDuration
}

type BreathingDetailedResponse = {
    commentary?: string,
    inhale?: BreathingCharactersticsResponse,
    exhale?: BreathingCharactersticsResponse,
    audio_params: AudioParams
}

type SpeechDetailedResponse = {
    audio_params: AudioParams,
    commentary?: string
}

type SpectrogramDatabaseResponse = {
    spectrogram_path: string
}

type PatchPatientDetailsRequest = {
    age?: number,
    gender?: string,
    is_smoking?: boolean,
}

type PatchStatusInfoRequest = {
    doctor_status?: string,
    marking_status?: string
}

type PutEpisodesRequest = {
    episodes: AudioEpisode[]
}

type PatchGeneralInfoRequest = PatchStatusInfoRequest & PatchPatientDetailsRequest & {
    is_visible?: boolean,
    covid19_symptomatic_type?: string
}

type PatchAudioParams = Partial<Pick<AudioParams, 'is_representative' | 'is_representative_scientist' 
                                                | 'is_validation_audio' | 'is_marked' | 'is_marked_scientist' 
                                                | 'noise_type'>>

type CoughCharacteristicsPatch = {
    audio_params?: PatchAudioParams
    productivity?: string,
    intensity?: string,
    symptom_duration?: number,
    commentary?: string
}

type SpeechCharacteristicsPatch = {
    commentary?: string,
    audio_params?: PatchAudioParams
}

type BreathingCharactersticsPatch = {
    commentary?: string,
    inhale?: BreathingCharactersticsResponse,
    exhale?: BreathingCharactersticsResponse,
    audio_params?: PatchAudioParams
}

const MARKING_RECORDS_SORTING_FIELDS = ['marking_status', 'doctor_status', 'covid_status', 
    'full_diagnosis', 'data_source', 'identifier', 'request_id', 'date_created',
    'is_marked_doctor_cough', 'is_marked_doctor_breathing', 'is_marked_doctor_speech', 
    'is_marked_scientist_cough', 'is_marked_scientist_breathing', 'is_marked_scientist_speech'];
const MARKING_RECORDS_REGEXP = createSortByRegExp(MARKING_RECORDS_SORTING_FIELDS);

export const getMarkingRecords = async (req: Request, res: Response) => {
    const paginationParams = req.paginationParams;
    const sortingParams = getSortingParamsByRegExp(req.query.sort_by as string, MARKING_RECORDS_REGEXP);

    try {
        const doctorStatusFilterParams = req.query.doctor_status;
        const markingStatusFilterParams = req.query.marking_status;
        const covidStatusFilterParams = req.query.covid_status;
        const sourceFilterParams = req.query.data_source;
        const doctorCoughFilterParams = req.query.is_marked_doctor_cough;
        const doctorBreathFilterParams = req.query.is_marked_doctor_breathing;
        const doctorSpeechFilterParams = req.query.is_marked_doctor_speech;
        const scientistCoughFilterParams = req.query.is_marked_scientist_cough;
        const scientistBreathFilterParams = req.query.is_marked_scientist_breathing;
        const scientistSpeechFilterParams = req.query.is_marked_scientist_speech;

        const doctorStatusFilter = new Array<DatasetMarkingStatus>();
        const markingStatusFilter = new Array<DatasetMarkingStatus>();
        const covidStatusFilter = new Array<Covid19SymptomaticTypes>();
        let sourceFilter: string[];
        let doctorCoughFilter: boolean;
        let doctorBreathFilter: boolean;
        let doctorSpeechFilter: boolean;
        let scientistCoughFilter: boolean;
        let scientistBreathFilter: boolean;
        let scientistSpeechFilter: boolean;
    
        if (Array.isArray(doctorStatusFilterParams)) {
            for (const doctorStatus of doctorStatusFilterParams) {
                const type = Object.values(DatasetMarkingStatus).find(val => val == doctorStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                doctorStatusFilter.push(type);
            }
        } else if (doctorStatusFilterParams != undefined) {
            const type = Object.values(DatasetMarkingStatus).find(val => val == doctorStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            doctorStatusFilter.push(type);
        }

        if (Array.isArray(markingStatusFilterParams)) {
            for (const markingStatus of markingStatusFilterParams) {
                const type = Object.values(DatasetMarkingStatus).find(val => val == markingStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                markingStatusFilter.push(type);
            }
        } else if (markingStatusFilterParams != undefined) {
            const type = Object.values(DatasetMarkingStatus).find(val => val == markingStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            markingStatusFilter.push(type);
        }

        if (Array.isArray(covidStatusFilterParams)) {
            for (const covidStatus of covidStatusFilterParams) {
                const type = Object.values(Covid19SymptomaticTypes).find(val => val == covidStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                covidStatusFilter.push(type);
            }
        } else if (covidStatusFilterParams != undefined) {
            const type = Object.values(Covid19SymptomaticTypes).find(val => val == covidStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            covidStatusFilter.push(type);
        }

        let patientList: string[];
        const userRepo = getCustomRepository(UserRepository);
        const doctorUser = await userRepo.findOne(req.token.userId);
        if (!doctorUser.is_all_patients) { 
            patientList = await userRepo.findPatientsByUserId(req.token.userId);
            if (patientList.length == 0) {
                const errorMessage = getErrorMessage(HttpErrors.NO_PATIENTS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            } 
        } 
        // patientList = [] if is_all_patients
        if (Array.isArray(sourceFilterParams)) {
            for (const source of sourceFilterParams) {
                const user = await userRepo.findByLogin(source as string);
                if (user == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                if (patientList.length >= 1 && !patientList.includes(source as string)) {
                    const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                sourceFilter.push(user.login);
            }
        } else if (sourceFilterParams != undefined) {
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findByLogin(sourceFilterParams as string);
            if (user == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            if (patientList.length >= 1 && !patientList.includes(sourceFilterParams as string)) {
                const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            sourceFilter.push(user.login);
        } else {
            sourceFilter = patientList;
        }

        if (doctorCoughFilterParams != undefined) {
            doctorCoughFilter = doctorCoughFilterParams == 'true';
        }
        if (doctorBreathFilterParams != undefined) {
            doctorBreathFilter = doctorBreathFilterParams == 'true';
        }
        if (doctorSpeechFilterParams != undefined) {
            doctorSpeechFilter = doctorSpeechFilterParams == 'true';
        }
        if (scientistCoughFilterParams != undefined) {
            scientistCoughFilter = scientistCoughFilterParams == 'true';
        }
        if (scientistBreathFilterParams != undefined) {
            scientistBreathFilter = scientistBreathFilterParams == 'true';
        }
        if (scientistSpeechFilterParams != undefined) {
            scientistSpeechFilter = scientistSpeechFilterParams == 'true';
        }

        const records = await fetchMarkingGeneral(paginationParams, sortingParams, {
            doctorStatusFilter: doctorStatusFilter,
            markingStatusFilter: markingStatusFilter,
            covidStatusFilter: covidStatusFilter,
            sourceFilter: sourceFilter,
            doctorCoughFilter: doctorCoughFilter,
            doctorBreathFilter: doctorBreathFilter,
            doctorSpeechFilter: doctorSpeechFilter,
            scientistCoughFilter: scientistCoughFilter,
            scientistBreathFilter: scientistBreathFilter,
            scientistSpeechFilter: scientistSpeechFilter
        });
        const response: MarkingResponse[] = records.map((record) => {
            const diagnosisName = getDiagnosisName({
                otherDisesaseName: record.other_disease_name,
                acuteType: record.acute_cough_type,
                chronicType: record.chronic_cough_type,
                disesaseType: record.disease_type,
            });
            
            return {
                date_created: record.date_created,
                doctor_status: record.doctor_status,
                marking_status: record.marking_status,
                full_diagnosis: diagnosisName,
                covid_status: record.covid_status ?? '-',
                request_id: record.request_id,
                data_source: record.data_source,
                identifier: record.identifier,
                is_marked_doctor_cough: record.is_marked_doctor_cough,
                is_marked_doctor_breathing: record.is_marked_doctor_breathing,
                is_marked_doctor_speech: record.is_marked_doctor_speech,
                is_marked_scientist_cough: record.is_marked_scientist_cough,
                is_marked_scientist_breathing: record.is_marked_scientist_breathing,
                is_marked_scientist_speech: record.is_marked_scientist_speech
            };
        });
        //if (sortingParams != undefined && sortingParams?.sortingColumn == 'full_diagnosis') {
        //    response.sort((a, b) =>  sortByString(a.full_diagnosis, b.full_diagnosis, sortingParams.sortingOrder));
        //}
        return res.status(HttpStatusCodes.SUCCESS).send(response);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getAudioGeneralInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    try {
        const connection = getConnection();

        const records = await connection.manager
            .createQueryBuilder(DatasetRequest, 'req')
            .select('req.id', 'request_id')
            .addSelect('user.login', 'data_source')
            .addSelect('details.age', 'age')
            .addSelect('gender_type.gender_type', 'gender')
            .addSelect('details.is_smoking', 'is_smoking')
            .addSelect('disease_type.disease_type', 'disease_type')
            .addSelect('acute_cough_types.acute_cough_types', 'acute_cough_type')
            .addSelect('chronic_cough_types.chronic_cough_type', 'chronic_cough_type')
            .addSelect('covid19_symptomatic_types.symptomatic_type', 'covid19_symptomatic_type')
            .addSelect('patient_diseases.other_disease_name', 'other_disease_name')
            .addSelect('marking_status.marking_status', 'marking_status')
            .addSelect('doctor_status.marking_status', 'doctor_status')
            .leftJoin('req.user', 'user')
            .leftJoin('req.patient_details', 'details')
            .leftJoin('details.gender', 'gender_type')
            .leftJoin('req.marking_status', 'marking_status')
            .leftJoin('req.doctor_status', 'doctor_status')
            .leftJoin('req.patient_diseases', 'patient_diseases')
            .leftJoin('patient_diseases.disease_type', 'disease_type')
            .leftJoin('patient_diseases.acute_cough_types', 'acute_cough_types')
            .leftJoin('patient_diseases.chronic_cough_types', 'chronic_cough_types')
            .leftJoin('patient_diseases.covid19_symptomatic_type', 'covid19_symptomatic_types')
            .where('req.id = :req_id', {req_id: requestId})
            .execute() as AudioGeneralInfoDatabaseResponse[];

        if (records.length != 1) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const dbResult = records[0];

        const diagnosisName = getDiagnosisName({
            otherDisesaseName: dbResult.other_disease_name,
            acuteType: dbResult.acute_cough_type,
            chronicType: dbResult.chronic_cough_type,
            disesaseType: dbResult.disease_type,
        });
        const response: AudioGeneralInfoResponse = {
            age: dbResult.age,
            gender: dbResult.gender,
            full_diagnosis: diagnosisName,
            covid19_symptomatic_type: dbResult.covid19_symptomatic_type,
            data_source: dbResult.data_source,
            marking_status: dbResult.marking_status,
            doctor_status: dbResult.doctor_status,
            is_smoking: dbResult.is_smoking,
            request_id: dbResult.request_id,
        };
        return res.status(HttpStatusCodes.SUCCESS).send(response);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getCoughDetailedInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    try {
        const audioInfoReq = fetchAudioParams(requestId, DatasetAudioTypes.COUGH);
        const record = fetchCoughDetailed(requestId);

        const dbResult = await Promise.all([
            audioInfoReq,
            record
        ]);
        const episodesRecords = dbResult[0];
        const coughData = dbResult[1];
        if (episodesRecords.length < 1 ) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const episodes: AudioEpisode[] = episodesRecords
            .filter((rec) => rec.episode_end != null && rec.episode_start != null)
            .map((rec) => {
                return {
                    start: rec.episode_start,
                    end: rec.episode_end,
                    id: rec.episode_id,
                    type: rec.episode_type
                };
            });

        const response: CoughDetailedResponse = {
            audio_params: {
                is_validation_audio: episodesRecords[0].is_validation_audio,
                is_representative: episodesRecords[0].is_representative,
                is_representative_scientist: episodesRecords[0].is_representative_scientist,
                samplerate: episodesRecords[0].samplerate,
                audio_path: episodesRecords[0].samplerate == undefined ? null :
                    `/v1/admin/marking/${requestId}/${DatasetAudioTypes.COUGH}/audio/`,
                spectrogram_path: episodesRecords[0].samplerate == undefined ? null :
                    `/v1/admin/marking/${requestId}/${DatasetAudioTypes.COUGH}/spectrogram/`,
                episodes: episodes,
                is_marked: episodesRecords[0].is_marked,
                is_marked_scientist: episodesRecords[0].is_marked_scientist,
                noise_type: episodesRecords[0].noise_type
            },
            productivity: coughData?.productivity,
            intensity: coughData?.intensity,
            commentary: coughData?.commentary,
            symptom_duration: coughData?.symptom_duration,
        };
        return res.status(HttpStatusCodes.SUCCESS).send(response);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getBreathingDetailedInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    try {
        const audioInfoReq = fetchAudioParams(requestId, DatasetAudioTypes.BREATHING);
        const breathingTypesReq = fetchBreathingDetailed(requestId);

        const dbResult = await Promise.all([audioInfoReq, breathingTypesReq]);
        const audioInfo = dbResult[0];
        const breathingTypes = dbResult[1];
        if (audioInfo.length < 1) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const inhaleData: BreathingCharactersticsResponse = breathingTypes
            .filter((type) => type.breathing_type == BreathingTypes.INHALE)
            .map((inhale) => {
                return {
                    depth_type: inhale.depth_type,
                    difficulty_type: inhale.difficulty_type,
                    duration_type: inhale.duration_type,
                };
            })[0];

        const exhaleData: BreathingCharactersticsResponse = breathingTypes
            .filter((type) => type.breathing_type == BreathingTypes.EXHALE)
            .map((exhale) => {
                return {
                    depth_type: exhale.depth_type,
                    difficulty_type: exhale.difficulty_type,
                    duration_type: exhale.duration_type,
                };
            })[0];

        const episodes: AudioEpisode[] = audioInfo
            .filter((rec) => rec.episode_end != null && rec.episode_start != null)
            .map((rec) => {
                return {
                    start: rec.episode_start,
                    end: rec.episode_end,
                    id: rec.episode_id,
                    type: rec.episode_type
                };
            });

        const response: BreathingDetailedResponse = {
            commentary: breathingTypes[0].commentary,
            inhale: inhaleData,
            exhale: exhaleData,
            audio_params: {
                is_validation_audio: audioInfo[0].is_validation_audio,
                is_representative: audioInfo[0].is_representative,
                is_representative_scientist: audioInfo[0].is_representative_scientist,
                samplerate: audioInfo[0].samplerate,
                audio_path: audioInfo[0].samplerate == undefined ? null
                    : `/v1/admin/marking/${requestId}/${DatasetAudioTypes.BREATHING}/audio/`,
                spectrogram_path: audioInfo[0].samplerate == undefined ? null
                    : `/v1/admin/marking/${requestId}/${DatasetAudioTypes.BREATHING}/spectrogram/`,
                episodes: episodes,
                is_marked: audioInfo[0].is_marked,
                is_marked_scientist: audioInfo[0].is_marked_scientist,
                noise_type: audioInfo[0].noise_type
            },
        };

        return res.status(HttpStatusCodes.SUCCESS).send(response);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getSpeechDetailedInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    try {
        const audioInfoReq = fetchAudioParams(requestId, DatasetAudioTypes.SPEECH);
        const record = fetchSpeechDetailed(requestId);

        const dbResult = await Promise.all([
            audioInfoReq,
            record
        ]);
        const episodesRecords = dbResult[0];
        const speechData = dbResult[1];

        if (episodesRecords.length < 1) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const episodes: AudioEpisode[] = episodesRecords
            .filter((rec) => rec.episode_end != null && rec.episode_start != null)
            .map((rec) => {
                return {
                    start: rec.episode_start,
                    end: rec.episode_end,
                    id: rec.episode_id,
                    type: rec.episode_type
                };
            });

        const response: SpeechDetailedResponse = {
            audio_params: {
                is_validation_audio: episodesRecords[0].is_validation_audio,
                is_representative: episodesRecords[0].is_representative,
                is_representative_scientist: episodesRecords[0].is_representative_scientist,
                samplerate: episodesRecords[0].samplerate,
                audio_path: episodesRecords[0].samplerate == undefined ? null 
                    : `/v1/admin/marking/${requestId}/${DatasetAudioTypes.SPEECH}/audio/`,
                spectrogram_path: episodesRecords[0] == undefined ? null
                    : `/v1/admin/marking/${requestId}/${DatasetAudioTypes.SPEECH}/spectrogram/`,
                episodes: episodes,
                is_marked: episodesRecords[0].is_marked,
                is_marked_scientist: episodesRecords[0].is_marked_scientist,
                noise_type: episodesRecords[0].noise_type
            },
            commentary: speechData?.commentary,
        };

        return res.status(HttpStatusCodes.SUCCESS).send(response);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getAudio = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);
    const type = req.params.type;

    const audioType = getTypeFromString(type);
    if (audioType == null) {
        const errorMessage = getErrorMessage(HttpErrors.NO_AUDIO_TYPE);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    try {
        const audioPath = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioPathByRequestId(requestId, audioType);

        if (audioPath == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        res.status(HttpStatusCodes.SUCCESS);
        const extension = getFileExtName(audioPath);
        res.setHeader('Content-Type', `audio/${extension}`);

        const stream = await fileService
            .getFileAsStream(audioPath);
         
        stream.on('error', (err) => {
            console.error(err);
            const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
            res.setHeader('Content-Type', 'application/json');
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        })
            .pipe(res);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const getSpectrogram = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);
    const type = req.params.type;

    const audioType = getTypeFromString(type);
    if (audioType == null) {
        const errorMessage = getErrorMessage(HttpErrors.NO_AUDIO_TYPE);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    try {
        const connection = getConnection();

        const spectrogramRes = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .select('audio.spectrogram_path', 'spectrogram_path')
            .leftJoin('audio.audio_type', 'audio_type')
            .where('audio.request_id = :req_id', {req_id: requestId})
            .andWhere('audio_type.audio_type = :type', {type: audioType})
            .execute() as SpectrogramDatabaseResponse[];

        if (spectrogramRes.length != 1) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const spectrogramPath = spectrogramRes[0].spectrogram_path;

        res.status(HttpStatusCodes.SUCCESS);
        res.setHeader('Content-Type', 'image/png');

        const stream = await fileService
            .getFileAsStream(spectrogramPath);
     
        stream.on('error', (err) => {
            console.error(err);
            const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
            res.setHeader('Content-Type', 'application/json');
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        })
            .pipe(res);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const patchGeneralInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    let requestBody: PatchGeneralInfoRequest;
    try {
        requestBody = {
            age: req.body.age,
            gender: req.body.gender,
            is_smoking: req.body.is_smoking,
            covid19_symptomatic_type: req.body.covid19_symptomatic_type,
            is_visible: req.body.is_visible,
            marking_status: req.body.marking_status,
            doctor_status: req.body.doctor_status,
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const connection = getConnection();

    let gender: GenderTypes;
    if (requestBody.gender != undefined) {
        try {
            gender = await getCustomRepository(GenderTypesRepository)
                .findByStringOrFail(requestBody.gender);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_GENDER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }

    let patientDetails: DatasetPatientDetails;
    try {
        patientDetails = await connection
            .manager
            .findOneOrFail(DatasetPatientDetails, {
                select: ['id', 'age', 'gender_type_id', 'is_smoking'],
                where: {request_id: requestId},
            });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    patientDetails.age = requestBody?.age ?? patientDetails.age;
    patientDetails.gender_type_id = gender?.id ?? patientDetails.gender_type_id;
    patientDetails.is_smoking = requestBody?.is_smoking ?? patientDetails.is_smoking;

    const updateEntities: Array<unknown> = [patientDetails];

    if (requestBody.covid19_symptomatic_type != undefined) {
        let covidType: Covid19Types;
        let patientDiseases: DatasetPatientDiseases;
        try {
            covidType = await getCustomRepository(CovidTypesRepository)
                .findByStringOrFail(requestBody.covid19_symptomatic_type);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        try {
            patientDiseases = await getCustomRepository(DatasetPatientDiseasesRepository)
                .findDiseasesByRequestIdOrFail(requestId);

            patientDiseases.covid19_symptomatic_type_id = covidType.id;
            updateEntities.push(patientDiseases);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }

    let patchedRequest: DatasetRequest | undefined;
    try {
        patchedRequest = await getPatchedRequestStatusInfo(requestBody.doctor_status, requestBody.marking_status, requestId);
    } catch (error) {
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    if (patchedRequest != undefined) {
        updateEntities.push(patchedRequest);
    }

    let datasetRequest: DatasetRequest;
    if (requestBody.is_visible != undefined) {
        try {
            datasetRequest = await connection
                .manager
                .findOneOrFail(DatasetRequest, {where: {id: requestId}});
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }

    if (datasetRequest != undefined) {
        datasetRequest.is_visible = requestBody?.is_visible ?? datasetRequest.is_visible;
        updateEntities.push(datasetRequest);
    }

    try {
        await connection.manager.save(updateEntities);
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const patchStatusInfo = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    let requestBody: PatchStatusInfoRequest;
    try {
        requestBody = {
            marking_status: req.body.marking_status,
            doctor_status: req.body.doctor_status,
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    let patchedRequest: DatasetRequest | undefined;
    try {
        patchedRequest = await getPatchedRequestStatusInfo(requestBody.doctor_status, requestBody.marking_status, requestId);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    if (patchedRequest != undefined) {
        try {
            await getConnection().manager.save(patchedRequest, {transaction: false});
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }
    return res.status(HttpStatusCodes.NO_CONTENT).send();
};

const getPatchedRequestStatusInfo = async (doctorStatus: string | undefined, markingStatus: string | undefined, requestId: number): Promise<DatasetRequest | undefined> => {
    const connection = getConnection();
    let doctorStatusEntity: EntityDatasetMarkingStatus;
    if (doctorStatus != undefined) {
        try {
            doctorStatusEntity = await connection
                .manager
                .findOneOrFail(EntityDatasetMarkingStatus, {where: {marking_status: doctorStatus}});
        } catch (error) {
            console.error(error);
            throw new Error(HttpErrors.NO_STATUS);
        }
    }

    let markingStatusEntity: EntityDatasetMarkingStatus;
    if (markingStatus != undefined) {
        try {
            markingStatusEntity = await connection
                .manager
                .findOneOrFail(EntityDatasetMarkingStatus, {where: {marking_status: markingStatus}});
        } catch (error) {
            console.error(error);
            throw new Error(HttpErrors.NO_STATUS);
        }
    }

    let datasetRequest: DatasetRequest;
    if (doctorStatus != undefined || markingStatus != undefined) {
        try {
            datasetRequest = await connection
                .manager
                .findOneOrFail(DatasetRequest, {where: {id: requestId}});
        } catch (error) {
            console.error(error);
            throw new Error(HttpErrors.NO_RECORD);
        }
    }

    if (datasetRequest != undefined) {
        datasetRequest.doctor_status = doctorStatusEntity ?? datasetRequest?.doctor_status;
        datasetRequest.marking_status = markingStatusEntity ?? datasetRequest?.marking_status;
    }
    return datasetRequest;
};

export const patchCoughCharacteristics = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);
    let requestBody: CoughCharacteristicsPatch;
    try {
        requestBody = {
            commentary: req.body.commentary,
            intensity: req.body.intensity,
            productivity: req.body.productivity,
            symptom_duration: req.body.symptom_duration,
            audio_params: req.body.audio_params
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const connection = getConnection();

    let intensity: CoughIntensityTypes;
    if (requestBody.intensity != undefined) {
        try {
            intensity = await connection
                .manager
                .findOneOrFail(CoughIntensityTypes, {where: {intensity_type: requestBody.intensity}});
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_CHARACTERISTIC);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }

    let productivity: CoughProductivityTypes;
    if (requestBody.productivity != undefined) {
        try {
            productivity = await connection
                .manager
                .findOneOrFail(CoughProductivityTypes, {where: {productivity_type: requestBody.productivity}});
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_CHARACTERISTIC);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    }

    let coughCharacteristics: DatasetCoughCharacteristics;
    try {
        coughCharacteristics = await connection
            .manager
            .findOneOrFail(DatasetCoughCharacteristics, {where: {request_id: requestId}});
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    coughCharacteristics.commentary = requestBody.commentary ?? coughCharacteristics.commentary;
    coughCharacteristics.intensity = intensity ?? coughCharacteristics.intensity;
    coughCharacteristics.productivity = productivity ?? coughCharacteristics.productivity;
    coughCharacteristics.symptom_duration = requestBody.symptom_duration ?? coughCharacteristics.symptom_duration;

    let updateEntities: Array<unknown> = [];
    updateEntities.push(coughCharacteristics);

    try {
        updateEntities = updateEntities.concat(await patchAudioParamsOrFail(requestBody.audio_params, requestId, DatasetAudioTypes.COUGH));
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    try {
        await connection.manager.save(updateEntities);
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const patchSpeechCharacteristics = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    let requestBody: SpeechCharacteristicsPatch;
    try {
        requestBody = {
            commentary: req.body.commentary,
            audio_params: req.body.audio_params
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const connection = getConnection();
    let speechCharacteristics: DatasetSpeechCharacteristics;
    try {
        speechCharacteristics = await connection
            .manager
            .findOne(DatasetSpeechCharacteristics, {where: {request_id: requestId}});
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }

    if (speechCharacteristics == undefined) {
        speechCharacteristics = new DatasetSpeechCharacteristics();
        speechCharacteristics.request_id = requestId;
    }

    speechCharacteristics.commentary = requestBody.commentary ?? speechCharacteristics.commentary;
    let updateEntities: Array<unknown> = [];
    updateEntities.push(speechCharacteristics);

    try {
        updateEntities = updateEntities.concat(await patchAudioParamsOrFail(requestBody.audio_params, requestId, DatasetAudioTypes.SPEECH));
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
   
    try {
        await connection.manager.save(updateEntities);
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.PATCH_SAVING_ERROR);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
};

const patchAudioParamsOrFail = async (audioParams: PatchAudioParams, requestId: number, type: DatasetAudioTypes) => {
    const result: Array<DatasetAudioInfo> = [];
    if (audioParams == undefined)
        return result;
    
    if (audioParams?.is_representative != undefined) {
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.is_representative = audioParams.is_representative;
        result.push(audioInfo);
    }

    if (audioParams?.is_representative_scientist != undefined) {
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.is_representative_scientist = audioParams.is_representative_scientist;
        result.push(audioInfo);
    }

    if (audioParams.is_validation_audio != undefined) {
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.is_validation_audio = audioParams.is_validation_audio;
        result.push(audioInfo);
    }    

    if (audioParams.is_marked != undefined) {
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.is_marked = audioParams.is_marked;
        result.push(audioInfo);
    }

    if (audioParams.is_marked_scientist != undefined) {
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.is_marked_scientist = audioParams.is_marked_scientist;
        result.push(audioInfo);
    }

    if (audioParams.noise_type != undefined) {
        let noiseType: NoiseTypes;
        const connection = getConnection();
        try {
            noiseType = await connection
                .manager
                .findOneOrFail(NoiseTypes, {where: {noise_type: audioParams.noise_type}});
        } catch (error) {
            console.error(error);
            throw new Error(HttpErrors.NO_NOISE_TYPE);
        }
        const audioInfo = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioInfoRequestByIdOrFail(requestId, type);
        audioInfo.noise_type = noiseType;
        result.push(audioInfo);
    }

    return result;    
};

export const patchBreathingCharacteristics = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    let requestBody: BreathingCharactersticsPatch;
    try {
        requestBody = {
            commentary: req.body.commentary,
            inhale: req.body.inhale,
            exhale: req.body.exhale,
            audio_params: req.body.audio_params
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const connection = getConnection();
    let datasetRequest: DatasetRequest;
    try {
        datasetRequest = await connection
            .manager
            .findOneOrFail(DatasetRequest, {
                relations: ['breathing_characteristics', 'breathing_characteristics.breathing_type',
                    'breathing_characteristics.depth_type', 'breathing_characteristics.difficulty_type',
                    'breathing_characteristics.duration_type'],
                where: {id: requestId},
            });
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    let inhale = datasetRequest
        ?.breathing_characteristics
        ?.find((characteristic) => characteristic.breathing_type.breathing_type == BreathingTypes.INHALE);

    let exhale = datasetRequest
        ?.breathing_characteristics
        ?.find((characteristic) => characteristic.breathing_type.breathing_type == BreathingTypes.EXHALE);

    const breathingTypesRepo = getCustomRepository(BreathingTypesRepository);
    const breathingDifficultyRepo = getCustomRepository(BreathingDifficultyTypesRepository);
    const breathingDepthRepo = getCustomRepository(BreathingDepthTypesRepository);
    const breathingDurationRepo = getCustomRepository(BreahtingDurationTypesRepository);

    let updateEntities: Array<unknown> = [];

    if (requestBody.inhale != undefined) {
        if (inhale == undefined) {
            inhale = new DatasetBreathingCharacteristics();
            try {
                inhale.breathing_type = await breathingTypesRepo.findByStringOrFail(BreathingTypes.INHALE);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }
            inhale.request_id = requestId;
        }

        let inhaleDepthType: EntityBreathingDepthTypes;
        if (requestBody.inhale.depth_type != undefined) {
            try {
                inhaleDepthType = await breathingDepthRepo.findByStringOrFail(requestBody.inhale.depth_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        let inhaleDifficultyType: EntityBreathingDifficultyTypes;
        if (requestBody.inhale.difficulty_type != undefined) {
            try {
                inhaleDifficultyType = await breathingDifficultyRepo.findByStringOrFail(requestBody.inhale.difficulty_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        let inhaleDurationType: EntityBreathingDurationTypes;
        if (requestBody.inhale.duration_type != undefined) {
            try {
                inhaleDurationType = await breathingDurationRepo.findByStringOrFail(requestBody.inhale.duration_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        inhale.depth_type = inhaleDepthType ?? inhale.depth_type;
        inhale.difficulty_type = inhaleDifficultyType ?? inhale.difficulty_type;
        inhale.duration_type = inhaleDurationType ?? inhale.duration_type;
        updateEntities.push(inhale);
    }

    if (requestBody.exhale != undefined) {
        if (exhale == undefined) {
            exhale = new DatasetBreathingCharacteristics();
            try {
                exhale.breathing_type = await breathingTypesRepo.findByStringOrFail(BreathingTypes.EXHALE);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            }
            exhale.request_id = requestId;
        }

        let exhaleDepthType: EntityBreathingDepthTypes;
        if (requestBody.exhale.depth_type != undefined) {
            try {
                exhaleDepthType = await breathingDepthRepo.findByStringOrFail(requestBody.exhale.depth_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }
        let exhaleDifficultyType: EntityBreathingDifficultyTypes;
        if (requestBody.exhale.difficulty_type != undefined) {
            try {
                exhaleDifficultyType = await breathingDifficultyRepo.findByStringOrFail(requestBody.exhale.difficulty_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        let exhaleDurationType: EntityBreathingDurationTypes;
        if (requestBody.exhale.duration_type != undefined) {
            try {
                exhaleDurationType = await breathingDurationRepo.findByStringOrFail(requestBody.exhale.duration_type);
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_BREATHING_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }

        exhale.depth_type = exhaleDepthType ?? exhale.depth_type;
        exhale.difficulty_type = exhaleDifficultyType ?? exhale.difficulty_type;
        exhale.duration_type = exhaleDurationType ?? exhale.duration_type;
        updateEntities.push(exhale);
    }

    try {
        updateEntities = updateEntities.concat(await patchAudioParamsOrFail(requestBody.audio_params, requestId, DatasetAudioTypes.BREATHING));
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    if (requestBody.commentary != undefined) {
        let breathingGeneralInfo = await connection
            .manager
            .findOne(DatasetBreathingGeneralInfo, {
                where: {request_id: requestId}
            });
        if (breathingGeneralInfo == undefined) {
            breathingGeneralInfo = new DatasetBreathingGeneralInfo();
            breathingGeneralInfo.request_id = requestId;
        }
        breathingGeneralInfo.commentary = requestBody.commentary;
        updateEntities.push(breathingGeneralInfo);
    }

    try {
        await connection.manager.save(updateEntities);
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const putAudioEpisodes = async (req: Request, res: Response) => {
    const type = req.params.type;
    const audioType = getTypeFromString(type);
    if (audioType == null) {
        const errorMessage = getErrorMessage(HttpErrors.NO_AUDIO_TYPE);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    
    if (audioType == DatasetAudioTypes.BREATHING) {
        return putBreathingAudioEpisodes(req, res);
    }
    const requestId = Number(req.params.id);

    let requestBody: PutEpisodesRequest;
    try {
        requestBody = {
            episodes: req.body.episodes,
        };
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const audioInfoRepo = getCustomRepository(DatasetAudioInfoRepository);
    let audioId: number;
    try {
        audioId = await audioInfoRepo.findAudioIdByRequestIdOrFail(requestId, type);
    } catch (error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }
    const queryRunner = getConnection().createQueryRunner();
    const newEpisodes = requestBody.episodes.filter(episode => episode.type == DomainDatasetEpisodesTypes.OTHER); // Prevent invalid values
    const manager = queryRunner.manager;
    try {
        await queryRunner.startTransaction();
        await updateEpisodes(manager, audioId, newEpisodes);
        await queryRunner.commitTransaction();
    } catch (error) {
        console.error(error);
        await queryRunner.rollbackTransaction();
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    } finally {
        await queryRunner.release();
    }
    return res.status(HttpStatusCodes.NO_CONTENT).send();
};

const putBreathingAudioEpisodes = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    let requestBody: PutEpisodesRequest;
    try {
        requestBody = {
            episodes: req.body.episodes
        };
        if (requestBody.episodes == undefined)
            throw new Error();
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    let audioId: number;
    try {
        audioId = await getCustomRepository(DatasetAudioInfoRepository)
            .findAudioIdByRequestIdOrFail(requestId, DatasetAudioTypes.BREATHING);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const inhaleEpisodes = requestBody
        .episodes
        .filter(episode => episode.type == DomainDatasetEpisodesTypes.BREATHING_INHALE);
    const exhaleEpisodes = requestBody
        .episodes
        .filter(episode => episode.type == DomainDatasetEpisodesTypes.BREATHING_EXHALE);

    const queryRunner = getConnection().createQueryRunner();
    await queryRunner.startTransaction();
    const manager = queryRunner.manager;

    try {
        if (inhaleEpisodes.length > 0) 
            await updateEpisodes(manager, audioId, inhaleEpisodes);
        
        if (exhaleEpisodes.length > 0) 
            await updateEpisodes(manager, audioId, exhaleEpisodes);

        await queryRunner.commitTransaction();
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    } catch(error) {
        console.error(error);
        await queryRunner.rollbackTransaction();
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    } finally {
        await queryRunner.release();
    }
}; 

const updateEpisodes = async (manager: EntityManager, audioId: number, newEpisodes: AudioEpisode[]) => {
    if (newEpisodes.length == 0)
        return;
    
    const type = Object
        .values(DomainDatasetEpisodesTypes)
        .find(val => val == newEpisodes[0].type);

    const episodeType = await manager
        .findOneOrFail(DatasetEpisodesTypes,{where: {episode_type: type}}) as DatasetEpisodesTypes;

    const episodesRepo = manager.getCustomRepository(DatasetAudioEpisodesRepository);
    const oldIds = await episodesRepo.findIdsByAudioIdAndType(audioId, type);
    if (oldIds.length > 0)
        await manager.getCustomRepository(DatasetAudioEpisodesRepository).delete(oldIds);
    
    const newEntities: DatasetAudioEpisodes[] = newEpisodes
        .map(episode => {
            const entity = new DatasetAudioEpisodes();
            entity.start = episode.start;
            entity.end = episode.end;
            entity.audio_info_id = audioId;
            entity.episode_type = episodeType;
            return entity;
        });
    await manager.save(newEntities, {transaction: false});
};

export const getNavigationById = async (req: Request, res: Response) => {
    const requestId = Number(req.params.id);

    const sortingParams = getSortingParamsByRegExp(req.query.sort_by as string, MARKING_RECORDS_REGEXP);
    try {
        const doctorStatusFilterParams = req.query.doctor_status;
        const markingStatusFilterParams = req.query.marking_status;
        const covidStatusFilterParams = req.query.covid_status;
        const sourceFilterParams = req.query.data_source;
        const doctorCoughFilterParams = req.query.is_marked_doctor_cough;
        const doctorBreathFilterParams = req.query.is_marked_doctor_breathing;
        const doctorSpeechFilterParams = req.query.is_marked_doctor_speech;
        const scientistCoughFilterParams = req.query.is_marked_scientist_cough;
        const scientistBreathFilterParams = req.query.is_marked_scientist_breathing;
        const scientistSpeechFilterParams = req.query.is_marked_scientist_speech;

        const doctorStatusFilter = new Array<DatasetMarkingStatus>();
        const markingStatusFilter = new Array<DatasetMarkingStatus>();
        const covidStatusFilter = new Array<Covid19SymptomaticTypes>();
        let sourceFilter: string[];
        let doctorCoughFilter: boolean;
        let doctorBreathFilter: boolean;
        let doctorSpeechFilter: boolean;
        let scientistCoughFilter: boolean;
        let scientistBreathFilter: boolean;
        let scientistSpeechFilter: boolean;
    
        if (Array.isArray(doctorStatusFilterParams)) {
            for (const doctorStatus of doctorStatusFilterParams) {
                const type = Object.values(DatasetMarkingStatus).find(val => val == doctorStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                doctorStatusFilter.push(type);
            }
        } else if (doctorStatusFilterParams != undefined) {
            const type = Object.values(DatasetMarkingStatus).find(val => val == doctorStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            doctorStatusFilter.push(type);
        }

        if (Array.isArray(markingStatusFilterParams)) {
            for (const markingStatus of markingStatusFilterParams) {
                const type = Object.values(DatasetMarkingStatus).find(val => val == markingStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                markingStatusFilter.push(type);
            }
        } else if (markingStatusFilterParams != undefined) {
            const type = Object.values(DatasetMarkingStatus).find(val => val == markingStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            markingStatusFilter.push(type);
        }

        if (Array.isArray(covidStatusFilterParams)) {
            for (const covidStatus of covidStatusFilterParams) {
                const type = Object.values(Covid19SymptomaticTypes).find(val => val == covidStatus);
                if (type == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                covidStatusFilter.push(type);
            }
        } else if (covidStatusFilterParams != undefined) {
            const type = Object.values(Covid19SymptomaticTypes).find(val => val == covidStatusFilterParams);
            if (type == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_COVID_STATUS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            covidStatusFilter.push(type);
        }

        let patientList: string[];
        const userRepo = getCustomRepository(UserRepository);
        const doctorUser = await userRepo.findOne(req.token.userId);
        if (!doctorUser.is_all_patients) { 
            patientList = await userRepo.findPatientsByUserId(req.token.userId);
            if (patientList.length == 0) {
                const errorMessage = getErrorMessage(HttpErrors.NO_PATIENTS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            } 
        } 
        // patientList = [] if is_all_patients
        if (Array.isArray(sourceFilterParams)) {
            for (const source of sourceFilterParams) {
                const user = await userRepo.findByLogin(source as string);
                if (user == undefined) {
                    const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                if (patientList.length >= 1 && !patientList.includes(source as string)) {
                    const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
                    return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
                }
                sourceFilter.push(user.login);
            }
        } else if (sourceFilterParams != undefined) {
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findByLogin(sourceFilterParams as string);
            if (user == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            if (patientList.length >= 1 && !patientList.includes(sourceFilterParams as string)) {
                const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
            sourceFilter.push(user.login);
        } else {
            sourceFilter = patientList;
        }

        if (doctorCoughFilterParams != undefined) {
            doctorCoughFilter = doctorCoughFilterParams == 'true';
        }
        if (doctorBreathFilterParams != undefined) {
            doctorBreathFilter = doctorBreathFilterParams == 'true';
        }
        if (doctorSpeechFilterParams != undefined) {
            doctorSpeechFilter = doctorSpeechFilterParams == 'true';
        }
        if (scientistCoughFilterParams != undefined) {
            scientistCoughFilter = scientistCoughFilterParams == 'true';
        }
        if (scientistBreathFilterParams != undefined) {
            scientistBreathFilter = scientistBreathFilterParams == 'true';
        }
        if (scientistSpeechFilterParams != undefined) {
            scientistSpeechFilter = scientistSpeechFilterParams == 'true';
        }

        const result = await fetchNavigationByRequestId(requestId, sortingParams, {
            doctorStatusFilter: doctorStatusFilter,
            markingStatusFilter: markingStatusFilter,
            covidStatusFilter: covidStatusFilter,
            sourceFilter: sourceFilter,
            doctorCoughFilter: doctorCoughFilter,
            doctorBreathFilter: doctorBreathFilter,
            doctorSpeechFilter: doctorSpeechFilter,
            scientistCoughFilter: scientistCoughFilter,
            scientistBreathFilter: scientistBreathFilter,
            scientistSpeechFilter: scientistSpeechFilter
        });
        if (result == undefined) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
        return res.status(HttpStatusCodes.SUCCESS).send(result);
    } catch(error) {
        console.error(error);
        const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
        return res.status(HttpStatusCodes.ERROR).send(errorMessage);
    }
};

export const deleteMarkingRecord = async (req: Request, res: Response) => {
    try {
        const requestId = Number(req.params.id);

        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;

        try {
            const coughAudioReq = manager.findOneOrFail(DatasetAudioInfo, {select: ['id', 'audio_path', 'spectrogram_path'], where: {request_id: requestId}});
            const audioIdsReq = manager.getCustomRepository(DatasetAudioInfoRepository).findAudioIdsByRequestId(requestId);

            const dbResult = await Promise.all([coughAudioReq, audioIdsReq]);
            const coughAudio = dbResult[0];
            const audioIds = dbResult[1];

            if (coughAudio?.spectrogram_path != undefined) {
                await fileService.deleteDirectory(coughAudio.spectrogram_path);
            }
            if (coughAudio?.audio_path != undefined) {
                await fileService.deleteDirectory(coughAudio.audio_path);
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
            
            await queryRunner.commitTransaction();
            return res.status(HttpStatusCodes.SUCCESS).send({
                is_deleted: true,
            });
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
};
