import {Request, Response} from 'express';
import {getCustomRepository} from 'typeorm';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {UserRepository} from '../infrastructure/repositories/userRepo';
import {doctorNotificationService} from '../container';

export class SupportController {
    public async requestSupportUnauthorized (req: Request, res: Response) {
        const contactData = String(req.body.email);
        const userMessage = String(req.body.message);
        if (!(contactData && userMessage)) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        try {
            await doctorNotificationService.notifyAboutSupportRequest(contactData, userMessage);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async requestSupportAuthorized (req: Request, res: Response) {
        const userId = Number(req.token.userId);
        const userMessage = String(req.body.message);
        if (!userMessage) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        try {
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findOne(userId);
            const contactData = user.email ?? user.phone_number;
            await doctorNotificationService.notifyAboutSupportRequest(contactData, userMessage, userId);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async requestDataDischarge (req: Request, res: Response) {
        const userId = Number(req.token.userId);
        try {
            const userRepo = getCustomRepository(UserRepository);
            const user = await userRepo.findOne(userId);
            const contactData = user.email ?? user.phone_number;
            user.is_active = false;
            await userRepo.save(user);
            await doctorNotificationService.notifyAboutSupportRequest(contactData, 'data_discharge', userId);
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }
}