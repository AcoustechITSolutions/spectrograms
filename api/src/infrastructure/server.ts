import  {Application} from 'express';
import morgan from 'morgan';
import healthCheckRoute from '../routes/healthCheckRoute';
import api_v1 from '../routes/v1';
import api_v1_1 from '../routes/v1.1';
import api_v1_2 from '../routes/v1.2';
import diagnosticBotRoute from '../routes/diagnosticBotRoute';
import newDiagnosticBotRoute from '../routes/newDiagnosticBotRoute';
import datasetBotRoute from '../routes/datasetBotRoute';
import muusBotRoute from '../routes/muusBotRoute';
import notifyDiagnosticBotRoute from '../routes/notifyDiagnosticBotRoute';
import paymentRoute from '../routes/paymentRoute';
import snsTopicsRoute from '../routes/snsTopicsRoute';
import helmet from 'helmet';
import {decodeToken} from '../domain/Session';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';

export const PORT: number = Number(process.env.API_PORT) || 3000;

export const setupServer = async (app: Application) => {
    morgan.token('user-id', function (req, res) {
        const token = req.headers['authorization'];
        const jwtPayload = decodeToken(token);
        if (!token || !jwtPayload) {
            return 'Unauthorized';
        }
        return `User ${jwtPayload.userId}:`;
    })
    app.use(morgan(':user-id [:date[clf]] ":method :url" :status :res[content-length] ":user-agent"'));
    app.use(helmet());

    const blockedSources = /(AID AI|Acoustery_corona_ai_diagnostics_edition)/;
    app.use(function(req, res, next) {
        const userAgent = req.headers['user-agent'];
        const isBlocked = userAgent?.match(blockedSources);
        if (!isBlocked) {
            next();
        } else {
            const errorMessage = getErrorMessage(HttpErrors.BLOCKED_SOURCE);
            return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
        }
    });
    
    app.use(api_v1);
    app.use(api_v1_1);
    app.use(api_v1_2);
    
    app.use('/', diagnosticBotRoute);
    app.use('/', newDiagnosticBotRoute);
    app.use('/', datasetBotRoute);
    app.use('/', muusBotRoute);
    app.use('/', notifyDiagnosticBotRoute);
    app.use('/', healthCheckRoute);
    app.use('/', paymentRoute);
    app.use('/', snsTopicsRoute);

    console.log(`node env ${process.env.NODE_ENV}`);
};
