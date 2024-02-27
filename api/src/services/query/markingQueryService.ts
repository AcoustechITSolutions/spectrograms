import {getConnection} from 'typeorm';
import {PaginationParams} from '../../interfaces/PaginationParams';
import {SortingParameters} from '../../interfaces/SortingParams';
import {DiseaseTypes, ChronicCoughTypes, AcuteCoughTypes} from '../../domain/DiseaseTypes';
import {Covid19SymptomaticTypes} from '../../domain/Covid19Types';
import {DatasetRequestStatus} from '../../domain/RequestStatus';
import {DatasetMarkingStatus} from '../../domain/DatasetMarkingStatus';
import {DatasetRequest} from '../../infrastructure/entity/DatasetRequest';
import {user} from '../../infrastructure/seed';
import {DatasetAudioTypes} from '../../domain/DatasetAudio';
import {BreathingTypes} from '../../domain/BreathingCharacteristics';
import {BreathingDepth} from '../../domain/BreathingCharacteristics';
import {BreathingDifficulty} from '../../domain/BreathingCharacteristics';
import {BreathingDuration} from '../../domain/BreathingCharacteristics';
import {DatasetSpeechCharacteristics} from '../../infrastructure/entity/DatasetSpeechCharacteristics';
import {DatasetAudioInfo} from '../../infrastructure/entity/DatasetAudioInfo';

export type MarkingDatabaseResponse = {
    readonly date_created: string,
    readonly file_path: string,
    readonly request_id: number,
    readonly marking_status: string,
    readonly doctor_status: string,
    readonly data_source: string,
    readonly identifier: string,
    readonly other_disease_name?: string,
    readonly disease_type: DiseaseTypes,
    readonly acute_cough_type?: AcuteCoughTypes,
    readonly chronic_cough_type?: ChronicCoughTypes,
    readonly covid_status?: Covid19SymptomaticTypes,
    readonly is_marked_doctor_cough: boolean,
    readonly is_marked_doctor_breathing: boolean,
    readonly is_marked_doctor_speech: boolean,
    readonly is_marked_scientist_cough: boolean,
    readonly is_marked_scientist_breathing: boolean,
    readonly is_marked_scientist_speech: boolean
}

export type AudioParamsDatabaseResponse = {
    readonly episode_start?: number,
    readonly episode_end?: number,
    readonly episode_id?: number,
    readonly episode_type?: string,
    readonly is_representative?: boolean,
    readonly is_representative_scientist?: boolean
    readonly is_validation_audio?: boolean,
    readonly samplerate: number,
    readonly is_marked: boolean,
    readonly is_marked_scientist: boolean,
    readonly noise_type: string
}

export type CoughDetailedDatabaseResponse = {
    readonly productivity: string,
    readonly intensity: string,
    readonly symptom_duration: number,
    readonly commentary: string,
}

export type BreathingDetailedTypesDatabaseResponse = {
    readonly breathing_type: BreathingTypes,
    readonly depth_type: BreathingDepth,
    readonly difficulty_type: BreathingDifficulty,
    readonly duration_type: BreathingDuration,
    readonly commentary?: string
}

export type SpeechDetailedDatabaseResponse = {
    readonly commentary?: string,
}

export type MarkingFilterParams = {
    doctorStatusFilter: DatasetMarkingStatus[],
    markingStatusFilter: DatasetMarkingStatus[],
    covidStatusFilter: Covid19SymptomaticTypes[],
    sourceFilter: string[],
    doctorCoughFilter?: boolean,
    doctorBreathFilter?: boolean,
    doctorSpeechFilter?: boolean,
    scientistCoughFilter?: boolean,
    scientistBreathFilter?: boolean,
    scientistSpeechFilter?: boolean
}

const getMarkingGeneralRequest = (filterParams?: MarkingFilterParams) => {
    const markingStatusQuery = getConnection()
        .manager
        .createQueryBuilder()
        .subQuery()
        .select('audio.request_id', 'request_id')
        .addSelect(`cast(min(case when audio.is_marked = false and types.audio_type = \'${DatasetAudioTypes.COUGH}\' then 0 else 1 end) as boolean)`, 'is_marked_doctor_cough')
        .addSelect(`cast(min(case when audio.is_marked = false and types.audio_type = \'${DatasetAudioTypes.BREATHING}\' then 0 else 1 end) as boolean)`, 'is_marked_doctor_breathing')
        .addSelect(`cast(min(case when audio.is_marked = false and types.audio_type = \'${DatasetAudioTypes.SPEECH}\' then 0 else 1 end) as boolean)`, 'is_marked_doctor_speech')
        .addSelect(`cast(min(case when audio.is_marked_scientist = false and types.audio_type = \'${DatasetAudioTypes.COUGH}\' then 0 else 1 end) as boolean)`, 'is_marked_scientist_cough')
        .addSelect(`cast(min(case when audio.is_marked_scientist = false and types.audio_type = \'${DatasetAudioTypes.BREATHING}\' then 0 else 1 end) as boolean)`, 'is_marked_scientist_breathing')
        .addSelect(`cast(min(case when audio.is_marked_scientist = false and types.audio_type = \'${DatasetAudioTypes.SPEECH}\' then 0 else 1 end) as boolean)`, 'is_marked_scientist_speech')
        .leftJoin('dataset_audio_types', 'types', 'types.id = audio.audio_type_id')
        .groupBy('audio.request_id')
        .from(DatasetAudioInfo, 'audio');

    const recordsQuery = getConnection()
        .manager
        .createQueryBuilder()
        .subQuery()
        .select('req.id', 'request_id')
        .addSelect('marking_status.marking_status', 'marking_status')
        .addSelect('doctor_status.marking_status', 'doctor_status')
        .addSelect('user.login', 'data_source')
        .addSelect('req.date_created', 'date_created')
        .addSelect('disease_type.disease_type', 'disease_type')
        .addSelect('acute_cough_types.acute_cough_types', 'acute_cough_type')
        .addSelect('chronic_cough_types.chronic_cough_type', 'chronic_cough_type')
        .addSelect('covid_type.symptomatic_type', 'covid_status')
        .addSelect('patient_diseases.other_disease_name', 'other_disease_name')
        .addSelect('patient_details.identifier', 'identifier')
        .addSelect('marking_data.is_marked_doctor_cough', 'is_marked_doctor_cough')
        .addSelect('marking_data.is_marked_doctor_breathing', 'is_marked_doctor_breathing')
        .addSelect('marking_data.is_marked_doctor_speech', 'is_marked_doctor_speech')
        .addSelect('marking_data.is_marked_scientist_cough', 'is_marked_scientist_cough')
        .addSelect('marking_data.is_marked_scientist_breathing', 'is_marked_scientist_breathing')
        .addSelect('marking_data.is_marked_scientist_speech', 'is_marked_scientist_speech')
        .leftJoin('dataset_marking_status', 'marking_status', 'req.marking_status_id = marking_status.id')
        .leftJoin('dataset_marking_status', 'doctor_status', 'req.doctor_status_id = doctor_status.id')
        .leftJoin('users', 'user', 'user.id = req.user_id')
        .leftJoin('dataset_request_status', 'req_status', 'req.status_id = req_status.id')
        .leftJoin('dataset_patient_diseases', 'patient_diseases', 'req.id = patient_diseases.request_id')
        .leftJoin('dataset_patient_details', 'patient_details', 'req.id = patient_details.request_id')
        .leftJoin('patient_diseases.disease_type', 'disease_type')
        .leftJoin('patient_diseases.acute_cough_types', 'acute_cough_types')
        .leftJoin('patient_diseases.chronic_cough_types', 'chronic_cough_types')
        .leftJoin('patient_diseases.covid19_symptomatic_type', 'covid_type')
        .leftJoin((subQuery) => {
            return markingStatusQuery;
        }, 'marking_data', 'marking_data.request_id = req.id')
        .where('req.is_visible = :visible', {visible: true})
        .andWhere('user.login != :user_login', {user_login: user.login})
        .andWhere('req_status.request_status = :status', {status: DatasetRequestStatus.PENDING})
        .from(DatasetRequest, 'req');

    if (filterParams?.doctorStatusFilter?.length >= 1) {
        recordsQuery.andWhere('doctor_status.marking_status IN (:...doctor_types)', {doctor_types: filterParams.doctorStatusFilter});
    }
    if (filterParams?.markingStatusFilter?.length >= 1) {
        recordsQuery.andWhere('marking_status.marking_status IN (:...marking_types)', {marking_types: filterParams.markingStatusFilter});
    }
    if (filterParams?.covidStatusFilter?.length >= 1) {
        recordsQuery.andWhere('covid_type.symptomatic_type IN (:...covid_types)', {covid_types: filterParams.covidStatusFilter});
    }
    if (filterParams?.sourceFilter?.length >= 1) {
        recordsQuery.andWhere('user.login IN (:...sources)', {sources: filterParams.sourceFilter});
    }
    
    if (filterParams?.doctorCoughFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_doctor_cough = :doctor_cough', {doctor_cough: filterParams.doctorCoughFilter});
    }
    if (filterParams?.doctorBreathFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_doctor_breathing = :doctor_breath', {doctor_breath: filterParams.doctorBreathFilter});
    }
    if (filterParams?.doctorSpeechFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_doctor_speech = :doctor_speech', {doctor_speech: filterParams.doctorSpeechFilter});
    }
    if (filterParams?.scientistCoughFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_scientist_cough = :scientist_cough', {scientist_cough: filterParams.scientistCoughFilter});
    }
    if (filterParams?.scientistBreathFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_scientist_breathing = :scientist_breath', {scientist_breath: filterParams.scientistBreathFilter});
    }
    if (filterParams?.scientistSpeechFilter != undefined) {
        recordsQuery
            .andWhere('marking_data.is_marked_scientist_speech = :scientist_speech', {scientist_speech: filterParams.scientistSpeechFilter});
    }
    
    return recordsQuery;
};

export const fetchMarkingGeneral = async (paginationParams: PaginationParams, sortingParams?: SortingParameters, filterParams?: MarkingFilterParams)
    : Promise<MarkingDatabaseResponse[]> => {
    const recordsQuery = getMarkingGeneralRequest(filterParams);

    if (sortingParams != undefined && sortingParams?.sortingColumn != 'full_diagnosis') {
        recordsQuery
            .orderBy(sortingParams.sortingColumn, sortingParams.sortingOrder);
    }
    if (sortingParams?.sortingColumn != 'date_created') {
        recordsQuery
            .addOrderBy('date_created', 'DESC');
    }
    const records = await recordsQuery
        .offset(paginationParams.offset)
        .limit(paginationParams.limit)
        .execute() as MarkingDatabaseResponse[];

    return records;
};

export const fetchAudioParams = async (requestId: number, type: DatasetAudioTypes) => {
    const connection = getConnection();
    return connection
        .createQueryBuilder(DatasetRequest, 'req')
        .select('audio.samplerate', 'samplerate')
        .addSelect('audio.is_marked', 'is_marked')
        .addSelect('audio.is_marked_scientist', 'is_marked_scientist')
        .addSelect('episodes.id', 'episode_id')
        .addSelect('episodes.start', 'episode_start')
        .addSelect('episodes.end', 'episode_end')
        .addSelect('episode_type.episode_type', 'episode_type')
        .addSelect('audio.is_representative', 'is_representative')
        .addSelect('audio.is_representative_scientist', 'is_representative_scientist')
        .addSelect('audio.is_validation_audio', 'is_validation_audio')
        .addSelect('noise_type.noise_type', 'noise_type')
        .leftJoin('req.audio_info', 'audio')
        .leftJoin('audio.episodes_duration', 'episodes')
        .leftJoin('audio.audio_type', 'audio_type')
        .leftJoin('audio.noise_type', 'noise_type')
        .leftJoin('episodes.episode_type', 'episode_type')
        .where('req.id = :req_id', {req_id: requestId})
        .andWhere('audio_type.audio_type = :type', {type: type})
        .execute() as Promise<AudioParamsDatabaseResponse[]>;
};

export const fetchCoughDetailed = async (requestId: number) => {
    const connection = getConnection();
    return connection
        .createQueryBuilder(DatasetRequest, 'req')
        .select('cough_productivity.productivity_type', 'productivity')
        .addSelect('cough_intensity.intensity_type', 'intensity')
        .addSelect('cough_char.symptom_duration', 'symptom_duration')
        .addSelect('cough_char.commentary', 'commentary')
        .leftJoin('req.cough_characteristics', 'cough_char')
        .leftJoin('cough_char.productivity', 'cough_productivity')
        .leftJoin('cough_char.intensity', 'cough_intensity')
        .where('req.id = :req_id', {req_id: requestId})
        .getRawOne() as Promise<CoughDetailedDatabaseResponse>;
};

export const fetchBreathingDetailed = async (requestId: number) => {
    const connection = getConnection();
    return connection
        .createQueryBuilder(DatasetRequest, 'req')
        .addSelect('duration_type.duration_type', 'duration_type')
        .addSelect('difficulty_type.difficulty_type', 'difficulty_type')
        .addSelect('depth_type.depth_type', 'depth_type')
        .addSelect('breathing_type.breathing_type', 'breathing_type')
        .addSelect('general.commentary', 'commentary')
        .leftJoin('req.breathing_characteristics', 'breathing')
        .leftJoin('req.breathing_general_info', 'general')
        .leftJoin('breathing.breathing_type', 'breathing_type')
        .leftJoin('breathing.depth_type', 'depth_type')
        .leftJoin('breathing.difficulty_type', 'difficulty_type')
        .leftJoin('breathing.duration_type', 'duration_type')
        .where('req.id = :req_id', {req_id: requestId})
        .execute() as Promise<BreathingDetailedTypesDatabaseResponse[]>;
};

export const fetchSpeechDetailed = async (requestId: number) => {
    const connection = getConnection();
    return connection
        .createQueryBuilder(DatasetSpeechCharacteristics, 'speech')
        .select('speech.commentary', 'commentary')
        .where('speech.request_id = :req_id', {req_id: requestId})
        .getRawOne() as Promise<SpeechDetailedDatabaseResponse>;
};

export const fetchNavigationByRequestId = async (requestId: number, sortingParams?: SortingParameters, filterParams?: MarkingFilterParams) => {
    let orderBy = '';
    if (sortingParams != undefined && sortingParams?.sortingColumn != 'full_diagnosis') {
        orderBy = `${sortingParams.sortingColumn} ${sortingParams.sortingOrder}`;
    }
    if (sortingParams?.sortingColumn != 'date_created' && orderBy.length > 0) {
        orderBy = orderBy.concat(', date_created desc');
    } else if (orderBy.length == 0) {
        orderBy = 'date_created desc';
    }

    const recordsQuery = getMarkingGeneralRequest(filterParams);
    return getConnection()
        .createQueryBuilder()
        .select('*')
        .from((subQuery) => {
            return subQuery
                .select('*')
                .select(`lag(request_id) over (order by ${orderBy})`, 'prev')
                .addSelect(`lead(request_id) over (order by ${orderBy})`, 'next')
                .addSelect('request_id')
                .from(_ => recordsQuery, 'x');
        }, 'data')
        .where(':req_id in ("request_id", "prev", "next")', {req_id: requestId})
        .andWhere('request_id = :req_id')
        .getRawOne();
};