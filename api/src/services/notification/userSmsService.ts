import {SNS} from 'aws-sdk'; 

export class SendUserSms {

    constructor(private sns: SNS) {}

    public async sendUserSms(destination: string, message: string) {
        try {
            const publishResult = await this.sns.publish({
                Message: message,
                PhoneNumber: destination
            }).promise();
            console.log(`SMS message id: ${publishResult.MessageId}`);
        } catch(error) {
            throw error;
        }
    }
}
