export enum DiseaseTypes {
    ACUTE = 'acute',
    CHRONIC = 'chronic',
    NONE = 'none'
}

type DiagnosisNameDTO = {
    disesaseType: DiseaseTypes,
    otherDisesaseName: string,
    acuteType?: AcuteCoughTypes,
    chronicType?: ChronicCoughTypes
}

export const getDiagnosisName = (dto: DiagnosisNameDTO): string => {
    const disesaseType = dto.disesaseType;
    let diagnosisName: string;
    switch (disesaseType) {
    case DiseaseTypes.NONE:
        diagnosisName = 'Заболевания отсутствуют'; // TODO: use i18n
        break;
    case DiseaseTypes.ACUTE:
        const acuteType = dto.acuteType;
        if (acuteType == AcuteCoughTypes.OTHER) {
            diagnosisName = dto.otherDisesaseName;
        } else {
            diagnosisName = acuteCoughTypesRu.get(acuteType);
        }
        break;
    case DiseaseTypes.CHRONIC:
        const chronicType = dto.chronicType;
        if (chronicType == ChronicCoughTypes.OTHER) {
            diagnosisName = dto.otherDisesaseName;
        } else {
            diagnosisName = chronicCoughTypesRu.get(chronicType);
        }
        break;
    }
    return diagnosisName;
};

export enum AcuteCoughTypes {
    OTHER = 'other', // Другое
    ACUTE_BRONCHITIS = 'acute_bronchitis', // Острый бронхит
    VIRAL_PNEUMONIA = 'viral_pneumonia', // Вирусная пневмония
    PLEURISY = 'pleurisy', // Плеврит
    PULMONARY_EMBOLISM = 'pulmonary_embolism', // ТЭЛА
    WHOOPING_COUGH = 'whooping_cough', // Коклюш
    PNEUMONIA = 'pneumonia', // Пневмония
    PNEUMONIA_COMPLICATION = 'pneumonia_complication', // Осложение пневмонии
}

export enum ChronicCoughTypes {
    OTHER = 'other', // Другое
    LUNG_INFARCTION = 'lung_infarction', // Инфаркт лёгкого
    BRONCHIAL_ASTHMA = 'bronchial_asthma', // Бронхиальная астма
    PSYCHOGENIC_COUGH = 'psychogenic_cough', // Психогенный кашель
    PRIMARY_TUBERCULOSIS_COMPLEX = 'primary_tuberculosis_complex', // Первичный туберкулёзный комплекс
    CHRONICAL_BRONCHITIS = 'chronical_bronchitis', // Хронический бронхит
    COPD = 'copd', // ХОБЛ
    BRONCHOECTATIC_DISEASE = 'bronchoectatic_disease', // Бронхоэктическая болезнь
    TUMORS = 'tumors', // Новообразования
    CONGESTIVE_HEART_FAILURE = 'congestive_heart_failure', // Застойная сердечная недостаточность
}

export const acuteCoughTypesRu = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Другое'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Острый бронхит'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Вирусная пневмония'],
    [AcuteCoughTypes.PLEURISY, 'Плеврит'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'ТЭЛА'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Коклюш'],
    [AcuteCoughTypes.PNEUMONIA, 'Пневмония'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Осложнение пневмонии'],
]);

export const acuteCoughTypesEn = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Other'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Acute bronchitis'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Viral pneumonia'],
    [AcuteCoughTypes.PLEURISY, 'Pleurisy'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Pulmonary embolism'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Pertussis'],
    [AcuteCoughTypes.PNEUMONIA, 'Pneumonia'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Pneumonia complication'],
]);

export const acuteCoughTypesSr = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Drugo'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Akutni bronhitis'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Virusna upala pluća'],
    [AcuteCoughTypes.PLEURISY, 'Pleurisija'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Plućna embolija'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Pertusis'],
    [AcuteCoughTypes.PNEUMONIA, 'Upala pluća'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Komplikacija upale pluća'],
]);

export const acuteCoughTypesKk = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Басқа'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Жедел бронхит'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Вирустық пневмония'],
    [AcuteCoughTypes.PLEURISY, 'Плеврит'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Өкпе эмболиясы'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Көкжөтел'],
    [AcuteCoughTypes.PNEUMONIA, 'Пневмония'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Пневмонияның асқынуы'],
]);

export const acuteCoughTypesVi = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Khác'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Viêm phế quản cấp'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Viêm phổi do virus'],
    [AcuteCoughTypes.PLEURISY, 'Viêm màng phổi'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Thuyên tắc phổi'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Ho gà'],
    [AcuteCoughTypes.PNEUMONIA, 'Viêm phổi'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Biến chứng viêm phổi'],
]);

export const acuteCoughTypesId = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Lain'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Bronkitis akut'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Pneumonia virus'],
    [AcuteCoughTypes.PLEURISY, 'Pleurisi'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Emboli paru'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Pertusis'],
    [AcuteCoughTypes.PNEUMONIA, 'Radang paru-paru'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Komplikasi pneumonia'],
]);

export const acuteCoughTypesFr = new Map<string, string>([
    [AcuteCoughTypes.OTHER, 'Autre'],
    [AcuteCoughTypes.ACUTE_BRONCHITIS, 'Bronchite aiguë'],
    [AcuteCoughTypes.VIRAL_PNEUMONIA, 'Pneumonie virale'],
    [AcuteCoughTypes.PLEURISY, 'Pleurésie'],
    [AcuteCoughTypes.PULMONARY_EMBOLISM, 'Embolie pulmonaire'],
    [AcuteCoughTypes.WHOOPING_COUGH, 'Coqueluche'],
    [AcuteCoughTypes.PNEUMONIA, 'Pneumonie'],
    [AcuteCoughTypes.PNEUMONIA_COMPLICATION, 'Complication de la pneumonie'],
]);

export const chronicCoughTypesRu = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Другое'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Инфаркт лёгкого'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Бронхиальная астма'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Психогенный кашель'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Первичный туберкулёзный комплекс'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Хронический бронхит'],
    [ChronicCoughTypes.COPD, 'ХОБЛ'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Бронхоэктическая болезнь'],
    [ChronicCoughTypes.TUMORS, 'Новообразования'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Застойная сердечная недостаточность'],
]);

export const chronicCoughTypesEn = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Other'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Lung infarction'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Bronchial asthma'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Psychogenic cough'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Primary tuberculosis complex'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Chronic bronchitis'],
    [ChronicCoughTypes.COPD, 'COPD'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Bronchiectatic disease'],
    [ChronicCoughTypes.TUMORS, 'Tumors'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Congestive heart failure'],
]);

export const chronicCoughTypesSr = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Drugo'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Infarkt pluća'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Bronhijalna astma'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Psihogeni kašalj'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Primarni kompleks tuberkuloze'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Hronični bronhitis'],
    [ChronicCoughTypes.COPD, 'HOBP'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Bronhiektazije'],
    [ChronicCoughTypes.TUMORS, 'Tumori'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Kongestivna srčana insuficijencija'],
]);

export const chronicCoughTypesKk = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Басқа'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Өкпе инфарктісі'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Бронх демікпесі'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Психогенді жөтел'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Бастапқы туберкулез кешені'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Созылмалы бронхит'],
    [ChronicCoughTypes.COPD, 'ӨСОА'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Бронхоэктатикалық ауру'],
    [ChronicCoughTypes.TUMORS, 'Ісіктер'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Жүректің тоқырауы'],
]);

export const chronicCoughTypesVi = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Khác'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Nhồi máu phổi'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Hen phế quản'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Ho do tâm lý'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Phức hợp bệnh lao nguyên phát'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Viêm phế quản mãn tính'],
    [ChronicCoughTypes.COPD, 'BPTNMT'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Bệnh giãn phế quản'],
    [ChronicCoughTypes.TUMORS, 'Khối u'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Suy tim sung huyết'],
]);

export const chronicCoughTypesId = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Lain'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Infark paru'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Asma bronkial'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Batuk psikogenik'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Kompleks tuberkulosis primer'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Bronkitis kronis'],
    [ChronicCoughTypes.COPD, 'PPOK'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Penyakit bronkiektatik'],
    [ChronicCoughTypes.TUMORS, 'Tumor'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Gagal jantung kongestif'],
]);

export const chronicCoughTypesFr = new Map<string, string>([
    [ChronicCoughTypes.OTHER, 'Autre'],
    [ChronicCoughTypes.LUNG_INFARCTION, 'Infarctus pulmonaire'],
    [ChronicCoughTypes.BRONCHIAL_ASTHMA, 'Asthme bronchique'],
    [ChronicCoughTypes.PSYCHOGENIC_COUGH, 'Toux psychogène'],
    [ChronicCoughTypes.PRIMARY_TUBERCULOSIS_COMPLEX, 'Complexe de tuberculose primaire'],
    [ChronicCoughTypes.CHRONICAL_BRONCHITIS, 'Bronchite chronique'],
    [ChronicCoughTypes.COPD, 'BPCO'],
    [ChronicCoughTypes.BRONCHOECTATIC_DISEASE, 'Bronchectasie'],
    [ChronicCoughTypes.TUMORS, 'Tumeurs'],
    [ChronicCoughTypes.CONGESTIVE_HEART_FAILURE, 'Insuffisance cardiaque congestive'],
]);