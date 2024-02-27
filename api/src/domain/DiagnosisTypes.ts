export enum DiagnosisTypes {
    HEALTHY = 'healthy',
    AT_RISK = 'at_risk',
    COVID_19 = 'covid_19'
}

export const diagnosisTypesRu = (diagnosis: string): string | undefined => {
    switch(diagnosis){
    case DiagnosisTypes.HEALTHY: return 'Здоров';
    case DiagnosisTypes.COVID_19: return 'COVID-19';
    case DiagnosisTypes.AT_RISK: return 'В зоне риска';
    default: return undefined;
    } 
};

export const fromMLDiagnosis = (diagnosis: string): DiagnosisTypes => {
    if (diagnosis == 'COVID-19') {
        return DiagnosisTypes.COVID_19;
    } else if (diagnosis == 'В зоне риска') {
        return DiagnosisTypes.AT_RISK;
    } else return DiagnosisTypes.HEALTHY;
};
