import {getErrorMessage, HttpStatusCodes, HttpErrors} from '../helpers/status';
import {Request, Response, NextFunction} from 'express';
import moment from 'moment';
import {
    Token,
    generateSignToken,
    verifyToken,
} from '../domain/Session';

export const verifyTokenMiddlewareDeprecated = async (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers['authorization'];

    if (!token) {
        const errorMessage = getErrorMessage(HttpErrors.NO_TOKEN);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }

    const jwtPayload = verifyToken(token);
    if (jwtPayload === undefined) {
        const errorMessage = getErrorMessage(HttpErrors.TOKEN_VERIFICATION_ERROR);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }
    
    const isJwtIdExpired: boolean = moment().isAfter(new Date(jwtPayload.exp * 1000));
    if (isJwtIdExpired) {
        const errorMessage = getErrorMessage(HttpErrors.TOKEN_EXPIRED);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }

    req.token = jwtPayload;

    const newToken = generateSignToken({
        userId: jwtPayload.userId, jwtId: jwtPayload.jwtId
    } as Token);
    res.setHeader('token', newToken);
    next();
};

export const verifyTokenMiddleware = async (req: Request, res: Response, next: NextFunction) => {
    const token = req.headers['authorization'];

    if (!token) {
        const errorMessage = getErrorMessage(HttpErrors.NO_TOKEN);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }

    const jwtPayload = verifyToken(token);
    if (jwtPayload === undefined) {
        const errorMessage = getErrorMessage(HttpErrors.TOKEN_VERIFICATION_ERROR);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }

    const isJwtIdExpired: boolean = moment().isAfter(new Date(jwtPayload.exp * 1000));
    if (isJwtIdExpired) {
        const errorMessage = getErrorMessage(HttpErrors.TOKEN_EXPIRED);
        return res.status(HttpStatusCodes.UNAUTHORIZED).send(errorMessage);
    }
    
    res.locals.jwtPayload = jwtPayload;
    req.token = jwtPayload;
    res.setHeader('token', token);
    next();
};
