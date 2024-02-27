import {getCustomRepository} from 'typeorm';
import {UserRepository} from '../../infrastructure/repositories/userRepo';
import {SNS} from 'aws-sdk'; 
import {DoctorNotificationService} from './DoctorNotificationService';

export class DoctorNotificationServiceSmsImpl extends DoctorNotificationService {

    constructor(private sns: SNS) {
        super();
    }

    public async notifyAboutNewDiagnostic() {
        try {
            const userRepo = getCustomRepository(UserRepository);
            const phoneNumbers = await userRepo.findDoctorsPhoneNumbers();
            for (const phoneNumber of phoneNumbers) {
                const publishResult = await this.sns.publish({
                    Message: this.NEW_DIAGNOSTIC_MESSAGE,
                    PhoneNumber: phoneNumber
                }).promise();
                console.log(`SMS message id: ${publishResult.MessageId}`);
            }
        } catch(error) {
            console.error(error);
        }
    }

    public async notifyAboutSupportRequest(contactData: string, userMessage: string, userId?: number) {
        try {
            const userRepo = getCustomRepository(UserRepository);
            const phoneNumbers = await userRepo.findDoctorsPhoneNumbers();
            for (const phoneNumber of phoneNumbers) {
                const supportMessage = `SUPPORT REQUEST\n\nUser id: ${userId ?? '-'}\nUser contact data: ${contactData}\nMessage: ${userMessage}`;
                const publishResult = await this.sns.publish({
                    Message: supportMessage,
                    PhoneNumber: phoneNumber
                }).promise();
                console.log(`SMS message id: ${publishResult.MessageId}`);
            }
        } catch(error) {
            console.error(error);
        }
    }
}
