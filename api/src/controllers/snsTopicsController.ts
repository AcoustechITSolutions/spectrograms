import {Request, Response} from 'express';
import {getCustomRepository} from 'typeorm';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {UserRepository} from '../infrastructure/repositories/userRepo';
import {notificationSns} from '../container';

export class SnsController {
    public async handleSesFailures (req: Request, res: Response) {
        if (req.headers['x-amz-sns-message-type'] === 'SubscriptionConfirmation') {
            const params = {
                Token: req.body.Token,
                TopicArn: req.body.TopicArn
            };
            notificationSns.confirmSubscription(params, function(err, data) {
                if (err) {
                    console.error(err);
                    throw err;
                } 
                console.log(data);
            });
            return res.status(HttpStatusCodes.NO_CONTENT).send();
            
        } else if (req.headers['x-amz-sns-message-type'] === 'Notification' && req.body.Message) {
            const message = JSON.parse(req.body.Message);
            if (message.notificationType == 'Bounce' || message.notificationType == 'Complaint') {
                if (message.mail && message.mail.destination) {
                    const address = message.mail.destination[0];
                    try {
                        const userRepo = getCustomRepository(UserRepository);
                        const user = await userRepo.findByEmailOrNumber(address.toLowerCase());
                        if (user == undefined) {
                            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
                            throw Error(errorMessage.error);
                        }
    
                        user.is_email_error = true;
                        user.email_error_type = message.notificationType;
    
                        await userRepo.save(user);
                    } catch (error) {
                        console.error(error.message);
                    }
                }
            }
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        }  
    }
}