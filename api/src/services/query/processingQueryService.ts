import {getConnection, SelectQueryBuilder} from 'typeorm';
import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {PaginationParams} from '../../interfaces/PaginationParams';
import {SortingParameters} from '../../interfaces/SortingParams';
import {Gender} from '../../domain/Gender';
import {DiagnosticReport} from '../../infrastructure/entity/DiagnosticReport';
import {DiagnosticRequest} from '../../infrastructure/entity/DiagnosticRequest';
import {DiagnosticRequestStatus as EntityDiagnosticRequestStatus} from '../../infrastructure/entity/DiagnostRequestStatus';
import {DiagnosisTypes} from '../../infrastructure/entity/DiagnosisTypes';
import {PatientInfo} from '../../infrastructure/entity/PatientInfo';
import {CoughAudio} from '../../infrastructure/entity/CoughAudio';
import {DiagnosisTypes as DomainDiagnosisTypes} from '../../domain/DiagnosisTypes';
import {CoughCharacteristics} from '../../infrastructure/entity/CoughCharacteristic';
import {TelegramDiagnosticRequest} from '../../infrastructure/entity/TelegramDiagnosticRequest';
import {NavigationParams} from '../../interfaces/NavigationParams';
import {User} from '../../infrastructure/entity/Users';
import {UserRoleTypes} from '../../domain/UserRoles';

export type ProcessingGeneralRecords = {
    readonly serial_number: number,
    readonly status: string,
    readonly diagnosis: string,
    readonly request_id: number,
    readonly date_created: Date,
    readonly age: number,
    readonly gender: string,
    readonly identifier: string,
    readonly nationality: string,
    readonly is_pcr_positive: boolean,
    readonly file_path: string
}

export interface AgeFilter {
    lte?: number,
    gte?: number
}

export type ProcessingFilterParams = {
    roleFilter: UserRoleTypes,
    sourceFilter: string[],
    genderFilter?: Gender[],
    statusFilter?: DiagnosticRequestStatus[],
    diagnosisFilter?: DomainDiagnosisTypes[],
    ageFilter?: AgeFilter
}

export interface ProcessingRecord {
    readonly intensity: string,
    readonly productivity: string,
    readonly diagnosis: string,
    readonly diagnosis_probability: number,
    readonly commentary: string,
    readonly request_id: number,
    readonly date_created: Date,
    readonly status: string,
    readonly language: string,
    readonly identifier: string,
    readonly nationality: string,
    readonly is_pcr_positive: boolean
}

export type StaticticResponse = {
    readonly category: DomainDiagnosisTypes,
    readonly value: number
}

export const getProcessingGeneralRequest = (filterParams?: ProcessingFilterParams) => {
    const processingRecordsQuery = getConnection()
        .manager
        .createQueryBuilder()
        .subQuery()
        .select('req.id', 'request_id')
        .addSelect('req_status.request_status', 'status')
        .addSelect('diagnosis.diagnosis_type', 'diagnosis')
        .addSelect('req."dateCreated"', 'date_created')
        .addSelect('user.login', 'data_source')
        .addSelect('patient.age', 'age')
        .addSelect('patient.gender', 'gender')
        .addSelect('patient.identifier', 'identifier')
        .addSelect('report.nationality', 'nationality')
        .addSelect('report.is_pcr_positive', 'is_pcr_positive')
        .addSelect('cough_audio.file_path', 'file_path')
        .leftJoin(User, 'user', 'user.id = req.user_id')
        .leftJoin(PatientInfo, 'patient', 'patient.request_id = req.id')
        .leftJoin(CoughAudio, 'cough_audio', 'cough_audio.request_id = req.id')
        .leftJoin(EntityDiagnosticRequestStatus, 'req_status', 'req_status.id = req.status_id')
        .leftJoin(DiagnosticReport, 'report', 'report.request_id = req.id')
        .leftJoin(DiagnosisTypes, 'diagnosis', 'diagnosis.id = report.diagnosis_id')
        .where('req_status.request_status NOT IN (:...types)', {types: [DiagnosticRequestStatus.PROCESSING, DiagnosticRequestStatus.ERROR]})
        .from(DiagnosticRequest, 'req');
    
    if (filterParams?.sourceFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('user.login IN (:...sources)', {sources: filterParams.sourceFilter});
    }
    if (filterParams?.ageFilter?.gte != undefined) {
        processingRecordsQuery
            .andWhere('patient.age >= :gte_age', {gte_age: filterParams.ageFilter.gte});
    }
    if (filterParams?.ageFilter?.lte != undefined) {
        processingRecordsQuery
            .andWhere('patient.age <= :lte_age', {lte_age: filterParams.ageFilter.lte});
    }
    if (filterParams?.genderFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('patient.gender IN (:...gender_types)', {gender_types: filterParams.genderFilter});
    }
    if (filterParams?.statusFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('req_status.request_status IN (:...status_types)', {status_types: filterParams.statusFilter});
    }
    if (filterParams?.diagnosisFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('diagnosis.diagnosis_type IN (:...diagnosis_types)', {diagnosis_types: filterParams.diagnosisFilter});
    }

    return processingRecordsQuery;
};

export const getProcessingViewerRequest = (filterParams?: ProcessingFilterParams) => {
    const processingRecordsQuery = getConnection()
        .manager
        .createQueryBuilder()
        .subQuery()
        .distinctOn(['patient.identifier'])
        .select('req.id', 'request_id')
        .addSelect('req_status.request_status', 'status')
        .addSelect('diagnosis.diagnosis_type', 'diagnosis')
        .addSelect('req."dateCreated"', 'date_created')
        .addSelect('user.login', 'data_source')
        .addSelect('patient.age', 'age')
        .addSelect('patient.gender', 'gender')
        .addSelect('patient.identifier', 'identifier')
        .addSelect('report.nationality', 'nationality')
        .addSelect('report.is_pcr_positive', 'is_pcr_positive')
        .addSelect('cough_audio.file_path', 'file_path')
        .leftJoin(User, 'user', 'user.id = req.user_id')
        .leftJoin(PatientInfo, 'patient', 'patient.request_id = req.id')
        .leftJoin(CoughAudio, 'cough_audio', 'cough_audio.request_id = req.id')
        .leftJoin(EntityDiagnosticRequestStatus, 'req_status', 'req_status.id = req.status_id')
        .leftJoin(DiagnosticReport, 'report', 'report.request_id = req.id')
        .leftJoin(DiagnosisTypes, 'diagnosis', 'diagnosis.id = report.diagnosis_id')
        .where('req_status.request_status NOT IN (:...types)', {types: [DiagnosticRequestStatus.PROCESSING, DiagnosticRequestStatus.ERROR]})
        .orderBy('patient.identifier')
        .addOrderBy('date_created', 'DESC')
        .from(DiagnosticRequest, 'req');
    
    if (filterParams?.sourceFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('user.login IN (:...sources)', {sources: filterParams.sourceFilter});
    }
    if (filterParams?.ageFilter?.gte != undefined) {
        processingRecordsQuery
            .andWhere('patient.age >= :gte_age', {gte_age: filterParams.ageFilter.gte});
    }
    if (filterParams?.ageFilter?.lte != undefined) {
        processingRecordsQuery
            .andWhere('patient.age <= :lte_age', {lte_age: filterParams.ageFilter.lte});
    }
    if (filterParams?.genderFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('patient.gender IN (:...gender_types)', {gender_types: filterParams.genderFilter});
    }
    if (filterParams?.statusFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('req_status.request_status IN (:...status_types)', {status_types: filterParams.statusFilter});
    }
    if (filterParams?.diagnosisFilter?.length >= 1) {
        processingRecordsQuery
            .andWhere('diagnosis.diagnosis_type IN (:...diagnosis_types)', {diagnosis_types: filterParams.diagnosisFilter});
    }

    return getConnection()
        .manager
        .createQueryBuilder()
        .subQuery()
        .select('*')
        .addSelect('row_number() over (order by data.date_created)', 'serial_number')
        .from((subQuery) => processingRecordsQuery, 'data');
};

export const fetchProcessingGeneral = async (paginationParams: PaginationParams, filterParams: ProcessingFilterParams, 
    sortingParams?: SortingParameters): Promise<ProcessingGeneralRecords[]> => {
    let processingRecordsQuery: SelectQueryBuilder<any>;
    if (filterParams?.roleFilter == UserRoleTypes.VIEWER) {
        processingRecordsQuery = getProcessingViewerRequest(filterParams);
    } else {
        processingRecordsQuery = getProcessingGeneralRequest(filterParams);
    }

    if (sortingParams != undefined && sortingParams.sortingColumn != 'date_created') {
        processingRecordsQuery
            .orderBy(sortingParams.sortingColumn, sortingParams.sortingOrder)
            .addOrderBy('date_created', 'DESC');
    } else if (sortingParams?.sortingColumn == 'date_created') {
        processingRecordsQuery
            .orderBy('date_created', sortingParams.sortingOrder);
    } else {
        processingRecordsQuery
            .orderBy('date_created', 'DESC'); // default case: order by date_created desc
    }

    const processingRecords = await processingRecordsQuery
        .offset(paginationParams.offset)
        .limit(paginationParams.limit)
        .execute() as ProcessingGeneralRecords[];
    return processingRecords;
};

export const fetchProcessingNavigationByRequestId = async (requestId: number, filterParams: ProcessingFilterParams,
    sortingParams?: SortingParameters): Promise<NavigationParams> => {
    let orderBy = '';
    if (sortingParams != undefined && sortingParams.sortingColumn != 'date_created') {
        orderBy = `${sortingParams.sortingColumn} ${sortingParams.sortingOrder}, date_created DESC`;
    } else if (sortingParams?.sortingColumn == 'date_created') {
        orderBy = `date_created ${sortingParams.sortingOrder}`;
    } else {
        orderBy = 'date_created desc'; // default case: order by date_created desc
    }
 
    let mainSubQuery: SelectQueryBuilder<any>;
    if (filterParams?.roleFilter == UserRoleTypes.VIEWER) {
        mainSubQuery = getProcessingViewerRequest(filterParams);
    } else {
        mainSubQuery = getProcessingGeneralRequest(filterParams);
    }
    const qb =  getConnection()
        .createQueryBuilder();
    return qb
        .select('*')
        .from((subQuery) => {
            return subQuery
                .select('*')
                .select(`lag(request_id) over (order by ${orderBy})`, 'prev')
                .addSelect(`lead(request_id) over (order by ${orderBy})`, 'next')
                .addSelect('request_id')
                .from(_ => mainSubQuery, 'x');
        }, 'data')
        .where(':req_id in ("request_id", "prev", "next")', {req_id: requestId})
        .andWhere('request_id = :req_id')
        .getRawOne() as Promise<NavigationParams>;
};

export const fetchProcessingById = async (requestId: number) => {    
    return getConnection()
        .createQueryBuilder()
        .select([
            'cough_char.intensity',
            'cough_char.productivity',
        ],
        )
        .addSelect('diagnosis_types.diagnosis_type', 'diagnosis')
        .addSelect('report.diagnosis_probability', 'diagnosis_probability')
        .addSelect('report.commentary', 'commentary')
        .addSelect('report.nationality', 'nationality')
        .addSelect('report.is_pcr_positive', 'is_pcr_positive')
        .addSelect('req.id', 'request_id')
        .addSelect('req."dateCreated"', 'date_created')
        .addSelect('req_status.request_status', 'status')
        .addSelect('req.language', 'language')
        .addSelect('patient.identifier', 'identifier')
        .from(DiagnosticReport, 'report')
        .leftJoin(DiagnosticRequest, 'req', 'req.id = report.request_id')
        .leftJoin(PatientInfo, 'patient', 'patient.request_id = req.id')
        .leftJoin(EntityDiagnosticRequestStatus, 'req_status', 'req_status.id = req.status_id')
        .leftJoin(TelegramDiagnosticRequest, 'tg', 'tg.request_id = req.id')
        .leftJoin(DiagnosisTypes, 'diagnosis_types', 'diagnosis_types.id = report.diagnosis_id')
        .leftJoin((subQuery) => {
            return subQuery
                .addSelect('cc.request_id', 'request_id')
                .addSelect('prod.productivity_type', 'productivity')
                .addSelect('intens.intensity_type', 'intensity')
                .from(CoughCharacteristics, 'cc')
                .leftJoin('cough_productivity_types', 'prod', 'cc.productivity_id = prod.id')
                .leftJoin('cough_intensity_types', 'intens', 'cc.intensity_id = intens.id');
        }, 'cough_char', 'req.id = cough_char.request_id')
        .where('req.id = :req_id', {req_id: requestId})
        .getRawOne() as Promise<ProcessingRecord>; 
};

export const fetchDiagnosisStatistics = async (filterParams: ProcessingFilterParams) => {
    let processingRecordsQuery: SelectQueryBuilder<any>;
    if (filterParams?.roleFilter == UserRoleTypes.VIEWER) {
        processingRecordsQuery = getProcessingViewerRequest(filterParams);
    } else {
        processingRecordsQuery = getProcessingGeneralRequest(filterParams);
    }

    return await getConnection()
        .manager
        .createQueryBuilder()
        .select('count(records.request_id)', 'value')
        .addSelect('records.diagnosis', 'category')
        .groupBy('records.diagnosis')
        .from((subQuery) => processingRecordsQuery, 'records')
        .execute() as StaticticResponse[];
}
