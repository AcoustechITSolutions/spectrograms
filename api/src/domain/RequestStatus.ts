export enum DiagnosticRequestStatus {
    PENDING = 'pending',
    ERROR = 'error',
    PROCESSING = 'processing',
    SUCCESS = 'success',
    NOISY_AUDIO = 'noisy_audio'
}

export enum TelegramDiagnosticRequestStatus {
    AGE = 'age',
    GENDER = 'gender',
    IS_SMOKING= 'is_smoking',
    IS_FORCED = 'is_forced',
    COUGH_AUDIO = 'cough_audio',
    DONE = 'done',
    CANCELLED = 'cancelled'
}

export enum TelegramDatasetRequestStatus {
    AGE = 'age',
    GENDER = 'gender',
    IS_SMOKING = 'is_smoking',
    IS_COVID = 'is_covid',
    IS_DISEASE = 'is_disease',
    DISEASE_NAME = 'disease_name',
    COUGH_AUDIO = 'cough_audio',
    IS_FORCED = 'is_forced',
    DONE = 'done',
    CANCELLED = 'cancelled'
}

export enum TgNewDiagnosticRequestStatus {
    START = 'start',
    SUPPORT = 'support',
    PAYMENT = 'payment',
    DISCLAIMER = 'disclaimer',
    CONDITIONS = 'conditions',
    AGE = 'age',
    GENDER = 'gender',
    IS_SMOKING= 'is_smoking',
    COUGH_AUDIO = 'cough_audio',
    DONE = 'done',
    CANCELLED = 'cancelled',
    LANGUAGE = 'language'
}

export enum HWRequestStatus {
    ERROR = 'error',
    PROCESSING = 'processing',
    SUCCESS = 'success'
}

export enum DatasetRequestStatus {
    PREPROCESSING = 'preprocessing',
    PREPROCESSING_ERROR = 'preprocessing_error',
    PENDING = 'pending',
    ERROR = 'error',
    DONE = 'done',
    CREATING_ERROR = 'creating_error',
}
