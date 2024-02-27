
import {getRepository} from 'typeorm';
import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {PaginationParams} from '../../interfaces/PaginationParams';
import {DiagnosticReport} from '../../infrastructure/entity/DiagnosticReport';
import {DiagnosticRequest} from '../../infrastructure/entity/DiagnosticRequest';

export type UserDatabaseReport = {
    readonly status: string,
    readonly date: Date,
    readonly request_id: number,
    readonly identifier?: string,
    readonly qr_code_token?: string,
    readonly probability?: number,
    readonly diagnosis?: string,
    readonly productivity?: string,
    readonly intensity?: string,
    readonly commentary?: string
}

export type QRDiagnosticReport = {
    readonly status: string,
    readonly date: Date,
    readonly request_id: number,
    readonly age?: number,
    readonly gender?: string,
    readonly identifier?: string,
    readonly probability?: number,
    readonly diagnosis?: string,
    readonly productivity?: string,
    readonly intensity?: string
}

export type HL7DiagnosticReport = {
    readonly date: Date,
    readonly status: DiagnosticRequestStatus,
    readonly diagnosis: string,
    readonly probability: number,
    readonly commentary: string,
    readonly user_id: number,
    readonly user_login: string,
    readonly age: number,
    readonly gender: string,
    readonly identifier: string,
    readonly nationality: string,
    readonly is_pcr_positive: boolean,
    readonly intensity: string,
    readonly productivity: string,
}

export type TelegramDiagnosticDatabaseReport = UserDatabaseReport & {tg_old_chat_id?: number, tg_new_chat_id?: number, muus_chat_id?: number, report_language?: string}

export const fetchDiagnosticByTelegramChatId = async (chatId: number, paginationParams: PaginationParams, userId: number) => {
    const userReports = await getGeneralTelegramDiagnosticQueryBuilder()
        .andWhere('(tg_req.chat_id = :chat_id OR tg_new_req.chat_id = :chat_id OR muus_req.chat_id = :chat_id)', {chat_id: chatId})
        .andWhere('req.user_id = :user_id', {user_id: userId})
        .orderBy('req.dateCreated', 'DESC')
        .offset(paginationParams.offset)
        .limit(paginationParams.limit)
        .execute() as UserDatabaseReport[];
        
    return userReports;
};

export const fetchDiagnosticWithChatIdByRequestId = async (requestId: number, userId: number): Promise<TelegramDiagnosticDatabaseReport | undefined> => {
    const userReports = await getGeneralTelegramDiagnosticQueryBuilder()
        .addSelect('tg_req.chat_id', 'tg_old_chat_id')
        .addSelect('tg_new_req.chat_id', 'tg_new_chat_id')
        .addSelect('muus_req.chat_id', 'muus_chat_id')
        .addSelect('req.language', 'report_language')
        .andWhere('req.id = :req_id', {req_id: requestId})
        .andWhere('req.user_id = :user_id', {user_id: userId})
        .execute() as TelegramDiagnosticDatabaseReport[];
    if (userReports.length > 0)
        return userReports[0];
    else
        return undefined;
};

const getGeneralTelegramDiagnosticQueryBuilder = () => {
    return getGeneralDiagnosticReportQueryBuilder()
        .leftJoin('tg_diagnostic_request', 'tg_req', 'tg_req.request_id=req.id')
        .leftJoin('tg_new_diagnostic_request', 'tg_new_req', 'tg_new_req.request_id=req.id')
        .leftJoin('muus_diagnostic_request', 'muus_req', 'muus_req.request_id=req.id');
};

export const fetchDiagnostic = async (
    userId: number, 
    paginationParams: PaginationParams, 
    byIds: number[] = undefined
): Promise<UserDatabaseReport[] | undefined> => {
    const basicQuery = getGeneralDiagnosticReportQueryBuilder()
        .andWhere('report.user_id = :user_id', {user_id: userId});

    if (byIds != undefined) {
        console.log(byIds);
        basicQuery
            .andWhere('req.id IN (:...ids_array)', {ids_array: byIds});
    }
    
    return await basicQuery
        .orderBy('req.dateCreated', 'DESC')
        .offset(paginationParams.offset)
        .limit(paginationParams.limit)
        .execute() as UserDatabaseReport[];
};

export const fetchDiagnosticById = async (
    userId: number, 
    requestId: number
): Promise<UserDatabaseReport | undefined> => {
    return await getGeneralDiagnosticReportQueryBuilder()
        .andWhere('report.user_id = :user_id', {user_id: userId})
        .andWhere('req.id = :req_id', {req_id: requestId})
        .getRawOne() as UserDatabaseReport;
};

export const fetchDiagnosticByQR = async ( 
    requestId: number,
    token: string
): Promise<QRDiagnosticReport | undefined> => {
    return await getGeneralDiagnosticReportQueryBuilder()
        .andWhere('report.qr_code_token = :qr_token', {qr_token: token})
        .andWhere('req.id = :req_id', {req_id: requestId})
        .getRawOne() as QRDiagnosticReport;
};

export const fetchHL7InfoById = async (
    userId: number, 
    requestId: number
): Promise<HL7DiagnosticReport | undefined> => {
    return await getGeneralDiagnosticReportQueryBuilder()
        .andWhere('report.user_id = :user_id', {user_id: userId})
        .andWhere('req.id = :req_id', {req_id: requestId})
        .getRawOne() as HL7DiagnosticReport;
};

export const fetchCoughAudioPath = async (
    userId: number,
    requestId: number
): Promise<string | undefined> => {
    const repo = getRepository(DiagnosticRequest);
    const {cough_audio} = await repo
        .createQueryBuilder('req')
        .select('audio.file_path', 'cough_audio')
        .leftJoin('cough_audio', 'audio', 'audio.request_id=req.id')
        .where('req.id = :req_id', {req_id: requestId})
        .andWhere('req.user_id = :user_id', {user_id: userId})
        .getRawOne();

    return cough_audio as string;
};

export const fetchSpectrogramPath = async (
    userId: number,
    requestId: number
): Promise<string | undefined> => {
    const repo = getRepository(DiagnosticRequest);
    const spectrogram = await repo
        .createQueryBuilder('req')
        .select('audio.spectrogram_path', 'spectrogram_path')
        .leftJoin('cough_audio', 'audio', 'audio.request_id=req.id')
        .where('req.id = :req_id', {req_id: requestId})
        .andWhere('req.user_id = :user_id', {user_id: userId})
        .getRawOne();
    
    return spectrogram.spectrogram_path;
};

const getGeneralDiagnosticReportQueryBuilder = () => {
    const diagnosticReportRepo = getRepository(DiagnosticReport);
    return diagnosticReportRepo
        .createQueryBuilder('report')
        .select('req.id', 'request_id')
        .addSelect('req.dateCreated', 'date')
        .addSelect('req_status.request_status', 'status')
        .addSelect('diagnosis_types.diagnosis_type', 'diagnosis')
        .addSelect('report.diagnosis_probability', 'probability')
        .addSelect('report.commentary', 'commentary')
        .addSelect('report.qr_code_token', 'qr_code_token')
        .addSelect('report.nationality', 'nationality')
        .addSelect('report.is_pcr_positive', 'is_pcr_positive')
        .addSelect('report.user_id', 'user_id')
        .addSelect('user.login', 'user_login')
        .addSelect('patient.age', 'age')
        .addSelect('patient.gender', 'gender')
        .addSelect('patient.identifier', 'identifier')
        .addSelect('cough_intensity.intensity_type', 'intensity')
        .addSelect('cough_productivity.productivity_type', 'productivity')
        .leftJoin('diagnostic_requests', 'req', 'report.request_id = req.id')
        .leftJoin('users', 'user', 'user.id = report.user_id')
        .leftJoin('patient_info', 'patient', 'report.request_id = patient.request_id')
        .leftJoin('diagnostic_request_status', 'req_status', 'req_status.id=req.status_id')
        .leftJoin('diagnosis_types', 'diagnosis_types', 'diagnosis_types.id=report.diagnosis_id')
        .leftJoin('cough_characteristics', 'cough', 'cough.request_id=req.id')
        .leftJoin('cough_intensity_types', 'cough_intensity', 'cough_intensity.id=cough.intensity_id')
        .leftJoin('cough_productivity_types', 'cough_productivity', 'cough_productivity.id=cough.productivity_id')
        .where('req_status.request_status != :processing_status',
            {processing_status: DiagnosticRequestStatus.PROCESSING})
        .andWhere('report.is_visible = :is_visible', {is_visible: true});
};
