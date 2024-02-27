export enum DatasetAudioTypes {
    BREATHING = 'breathing',
    COUGH = 'cough',
    SPEECH = 'speech'
}

export const getTypeFromString = (type: string): DatasetAudioTypes | null => {
    const index = Object.keys(DatasetAudioTypes).indexOf(type.toUpperCase());
    if (index < 0) {
        return null;
    } else {
        return Object.values(DatasetAudioTypes)[index];
    }
};
