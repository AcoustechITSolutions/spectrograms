export type ResponseMessage = {
    status: 'success' | HttpErrors
}

export interface SuccessMessage extends ResponseMessage {
    status: 'success'
}

export interface ErrorMessage extends ResponseMessage {
    status: HttpErrors,
    error?: string
}

const successMessage: ResponseMessage = {status: 'success'};
const getErrorMessage = (status: HttpErrors): ErrorMessage => {
    console.error(errorDescription.get(status));
    return {
        status: status,
        error: errorDescription.get(status)
    };
};

const enum HttpErrors {
    INTERNAL_ERROR = 'internal_error',
    INCORRECT_BODY = 'incorrect_body',
    NO_RECORD = 'no_record',
    FORBIDDEN = 'forbidden',
    BLOCKED_SOURCE = 'blocked_source',
    NO_USER = 'no_user',
    WRONG_VERIFICATION_CODE = 'wrong_verification_code',
    LOGIN_TAKEN = 'login_taken',
    PASSWORD_MISMATCH = 'password_mismatch',
    NO_LANGUAGE = 'no_language',
    NO_TOKEN = 'no_token',
    NO_REFRESH_TOKEN = 'no_refresh_token',
    TOKEN_VERIFICATION_ERROR = 'token_verification_error',
    TOKEN_EXPIRED = 'token_expired',
    REFRESH_TOKEN_VERIFICATION_ERROR = 'refresh_token_verification_error',
    REFRESH_TOKEN_EXPIRED = 'refresh_token_expired',
    NO_PATIENTS = 'no_patients',
    NO_ROLE = 'no_role',
    INVALID_NUMBER = 'invalid_number',
    WRONG_FORMAT = 'wrong_format',
    MESSAGE_SENDING_ERROR = 'message_sending_error',
    NO_GENDER = 'no_gender',
    NO_COVID_STATUS = 'no_covid_status',
    NO_DISEASE_TYPE = 'no_disease_type',
    NO_DISEASE_NAME = 'no_disease_name',
    FORBIDDEN_TO_DELETE = 'forbidden_to_delete',
    COUGH_NOT_DETECTED = 'cough_not_detected',
    NOT_ENOUGH_COUGH = 'not_enough_cough',
    NOT_ENOUGH_SPEECH = 'not_enough_speech',
    NOISY = 'noisy',
    NO_CHAT_ID = 'no_chat_id',
    NOT_NOISY = 'not_noisy',
    NO_SUBSCRIPTION = 'no_subscription',
    NO_DIAGNOSTICS = 'no_diagnostics',
    NO_FILES = 'no_files',
    NO_COUGH_FILE = 'no_cough_file',
    NO_SPEECH_FILE = 'no_speech_file',
    FILE_NAME_ERROR = 'file_name_error',
    FILE_FORMAT_ERROR = 'file_format_error',
    FILE_SENDING_ERROR = 'file_sending_error',
    FILE_SAVING_ERROR = 'file_saving_error', 
    NO_STATUS = 'no_status',
    NO_AUDIO_TYPE = 'no_audio_type',
    NO_COUGH_CHARACTERISTIC = 'no_cough_characteristic',
    NO_BREATHING_CHARACTERISTIC = 'no_breathing_characteristic',
    NO_DIAGNOSIS = 'no_diagnosis',
    PATCH_SAVING_ERROR = 'patch_saving_error',
    PATCH_RECORD_ERROR = 'patch_record_error',
    NO_PERSONAL_DATA = 'no_personal_data',
    NO_NOISE_TYPE = 'no_noise_type',
    HL7_GENERATION_ERROR = 'hl7_generation_error'
}

export const errorDescription = new Map<HttpErrors, string>([
    [HttpErrors.INTERNAL_ERROR, 'Internal server error'],
    [HttpErrors.INCORRECT_BODY, 'Request body missing or invalid'],
    [HttpErrors.NO_RECORD, 'No such record'],
    [HttpErrors.FORBIDDEN, 'User with this role does not have permission to view this resource'],
    [HttpErrors.BLOCKED_SOURCE, 'This application is blocked from sending requests'],
    [HttpErrors.NO_USER, 'No such user'],
    [HttpErrors.WRONG_VERIFICATION_CODE, 'Wrong verification code'],
    [HttpErrors.LOGIN_TAKEN, 'A user with this login or contact data already exists'], 
    [HttpErrors.PASSWORD_MISMATCH, 'Passwords do not match'],
    [HttpErrors.NO_LANGUAGE, 'Language not provided or invalid'],
    [HttpErrors.NO_TOKEN, 'Jwt-token not provided'],
    [HttpErrors.NO_REFRESH_TOKEN, 'Refresh token not provided'],
    [HttpErrors.TOKEN_VERIFICATION_ERROR, 'Jwt-token verification error'],
    [HttpErrors.TOKEN_EXPIRED, 'Jwt-token has expired'], 
    [HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR, 'Refresh token verification error'],
    [HttpErrors.REFRESH_TOKEN_EXPIRED, 'Refresh token has expired'], 
    [HttpErrors.NO_PATIENTS, 'This user can not have patients'],
    [HttpErrors.NO_ROLE, 'No such user role'],
    [HttpErrors.INVALID_NUMBER, 'Invalid mobile phone number, Amazon number validation failed'],
    [HttpErrors.WRONG_FORMAT, 'Email or phone number entered incorrectly. Number must be in international format, e.g. +71234567890'],
    [HttpErrors.MESSAGE_SENDING_ERROR, 'Unable to send messages to this contact data'],
    [HttpErrors.NO_GENDER, 'No such gender'],
    [HttpErrors.NO_COVID_STATUS, 'No such Covid-19 symptomatic type'],
    [HttpErrors.NO_DISEASE_TYPE, 'No such disease type'],
    [HttpErrors.NO_DISEASE_NAME, 'No such disease name'],
    [HttpErrors.FORBIDDEN_TO_DELETE, 'This user is not allowed to delete this record'],
    [HttpErrors.COUGH_NOT_DETECTED, 'Cough was not detected in the record'],
    [HttpErrors.NOT_ENOUGH_COUGH, 'Not enough cough sounds detected in the record'],
    [HttpErrors.NOT_ENOUGH_SPEECH, 'Not enough speech sounds detected in the record'],
    [HttpErrors.NOISY, 'Too much background noise in the record'],
    [HttpErrors.NO_CHAT_ID, 'No such Telegram chat id'],
    [HttpErrors.NOT_NOISY, 'Only noisy audio can be updated'],
    [HttpErrors.NO_SUBSCRIPTION, 'No subscriptions available for the user'],
    [HttpErrors.NO_DIAGNOSTICS, 'No available diagnostics left'],
    [HttpErrors.NO_FILES, 'No files were provided'],
    [HttpErrors.NO_COUGH_FILE, 'Cough audio was not provided'],
    [HttpErrors.NO_SPEECH_FILE, 'Speech audio was not provided'],
    [HttpErrors.FILE_NAME_ERROR, 'File names must be different'],
    [HttpErrors.FILE_FORMAT_ERROR, 'Unsupported audio format'],
    [HttpErrors.FILE_SENDING_ERROR, 'Unable to send file'],
    [HttpErrors.FILE_SAVING_ERROR, 'Unable to save file'],
    [HttpErrors.NO_STATUS, 'No such status'],
    [HttpErrors.NO_AUDIO_TYPE, 'No such audio type'],
    [HttpErrors.NO_COUGH_CHARACTERISTIC, 'No such intensity or productivity type'],
    [HttpErrors.NO_BREATHING_CHARACTERISTIC, 'No such depth, difficulty or duration type'],
    [HttpErrors.NO_DIAGNOSIS, 'No such diagnosis'],
    [HttpErrors.PATCH_SAVING_ERROR, 'Unable to save your data. Try later or check your request'],
    [HttpErrors.PATCH_RECORD_ERROR, 'Cannot patch request with error or noisy status'],
    [HttpErrors.NO_PERSONAL_DATA, 'No personal data provided for this user'],
    [HttpErrors.NO_NOISE_TYPE, 'No such noise type'],
    [HttpErrors.HL7_GENERATION_ERROR, 'Error while generating or saving HL7 message']
]);

const enum HttpStatusCodes {
    SUCCESS = 200,
    ERROR = 500,
    NOT_FOUND = 404,
    UNAUTHORIZED = 401,
    CONFLICT = 409,
    CREATED = 201,
    BAD_REQUEST = 400,
    NO_CONTENT = 204,
    FORBIDDEN = 403
}

export {
    HttpStatusCodes,
    successMessage,
    getErrorMessage,
    HttpErrors,
};
