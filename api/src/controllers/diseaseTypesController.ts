import {Request, Response} from 'express';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {DiseaseTypes,
    chronicCoughTypesRu, acuteCoughTypesRu, acuteCoughTypesEn, chronicCoughTypesEn,
    acuteCoughTypesSr, chronicCoughTypesSr, acuteCoughTypesKk, 
    chronicCoughTypesKk, acuteCoughTypesVi, chronicCoughTypesVi,
    acuteCoughTypesId, chronicCoughTypesId, acuteCoughTypesFr, 
    chronicCoughTypesFr} from '../domain/DiseaseTypes';

export const getDiseaseTypes = async (req: Request, res: Response) => {
    res.setHeader('Content-Type', 'application/json');
    return res.status(HttpStatusCodes.SUCCESS).send(JSON.stringify(
        Object.values(DiseaseTypes),
    ));
};

interface CoughTypes {
    key: string,
    value: string
}

export const getAcuteCoughTypes = async (req: Request, res: Response) => {
    let coughTypes: Map<string, string>;
    switch (req.query.lang) {
    case 'ru': {
        coughTypes = acuteCoughTypesRu;
        break;
    }
    case 'en': {
        coughTypes = acuteCoughTypesEn;
        break;
    }
    case 'sr': {
        coughTypes = acuteCoughTypesSr;
        break;
    }
    case 'vi': {
        coughTypes = acuteCoughTypesVi;
        break;
    }
    case 'kk': {
        coughTypes = acuteCoughTypesKk;
        break;
    }
    case 'id': {
        coughTypes = acuteCoughTypesId;
        break;
    }
    case 'fr': {
        coughTypes = acuteCoughTypesFr;
        break;
    }
    default: coughTypes = null;
    }

    if (coughTypes == null) {
        const errorMessage = getErrorMessage(HttpErrors.NO_LANGUAGE);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const response = new Array<CoughTypes>();
    coughTypes.forEach((val, key, map) => {
        response.push({key: key, value: val});
    });
    res.setHeader('Content-Type', 'application/json');
    return res.status(HttpStatusCodes.SUCCESS).send(response);
};

export const getChronicCoughTypes = async (req: Request, res: Response) => {
    let coughTypes: Map<string, string>;
    switch (req.query.lang) {
    case 'ru': {
        coughTypes = chronicCoughTypesRu;
        break;
    }
    case 'en': {
        coughTypes = chronicCoughTypesEn;
        break;
    }
    case 'sr': {
        coughTypes = chronicCoughTypesSr;
        break;
    }
    case 'kk': {
        coughTypes = chronicCoughTypesKk;
        break;
    }
    case 'vi': {
        coughTypes = chronicCoughTypesVi;
        break;
    }
    case 'id': {
        coughTypes = chronicCoughTypesId;
        break;
    }
    case 'fr': {
        coughTypes = chronicCoughTypesFr;
        break;
    }
    default: coughTypes = null;
    }

    if (coughTypes == null) {
        const errorMessage = getErrorMessage(HttpErrors.NO_LANGUAGE);
        return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
    }

    const response = new Array<CoughTypes>();
    coughTypes.forEach((val, key, map) => {
        response.push({key: key, value: val});
    });

    res.setHeader('Content-Type', 'application/json');
    return res.status(HttpStatusCodes.SUCCESS).send(response);
};
