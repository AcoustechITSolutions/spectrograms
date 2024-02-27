// export enum CoughIntensity {
//     INTENSIVE = 'intensive',
//     NOT_INTENSIVE = 'not_intensive',
//     WEAK = 'weak'
// }

export enum CoughIntensity {
    PAROXYSMAL = 'paroxysmal', // приступообразный
    PAROXYSMAL_HACKING = 'paroxysmal_hacking', // приступообразный, надсадный
    NOT_PAROXYSMAL = 'not_paroxysmal' // не приступообзаный
}

export enum CoughProductivity {
    PRODUCTIVE = 'productive',
    UNPRODUCTIVE = 'unproductive',
    WET_PRODUCTIVE_SMALL = 'wet_productive_small',
    DRY_PRODUCTIVE_SMALL = 'dry_productive_small'
}

export const coughQualityRu = (quality: string) => {
    switch(quality) {
    case CoughProductivity.DRY_PRODUCTIVE_SMALL: return 'Cухой/малопродуктивный';
    case CoughProductivity.WET_PRODUCTIVE_SMALL: return 'Влажный/малопродуктивный';
    case CoughProductivity.UNPRODUCTIVE: return 'Непродуктивный';
    case CoughProductivity.PRODUCTIVE: return 'Продуктивный';
    }
};

export const coughIntensityRu = (intensity: string) => {
    switch(intensity) {
    case CoughIntensity.NOT_PAROXYSMAL: return 'Не приступообразный';
    case CoughIntensity.PAROXYSMAL: return 'Приступообразный';
    case CoughIntensity.PAROXYSMAL_HACKING: return 'Приступообразный, надсадный';
    }
};
