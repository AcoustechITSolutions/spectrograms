import {SES} from 'aws-sdk'; 

export class SendUserEmail {

    constructor(private ses: SES) {}

    public async sendUserEmail(destination: string, message: string, subject: string) {
        try {
            const params = {
                Destination: { 
                    ToAddresses: [destination]
                },
                Message: { 
                    Body: { 
                        Html: {
                            Charset: 'UTF-8',
                            Data: message
                        }
                    },
                    Subject: {
                        Charset: 'UTF-8',
                        Data: subject
                    }
                },
                ReturnPath: process.env.SENDER_EMAIL,
                Source: process.env.SENDER_EMAIL
            };

            const publishResult = await this.ses.sendEmail(params).promise();
            console.log(`Email message id: ${publishResult.MessageId}`);
        } catch(error) {
            throw error;
        }
    }
}