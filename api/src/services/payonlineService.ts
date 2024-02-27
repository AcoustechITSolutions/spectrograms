/* eslint @typescript-eslint/no-var-requires: "off" */
const payOnline = require('payonline');
const payOnlineClient = new payOnline({
    merchantId: process.env.MERCHANT_ID,
    privateSecurityKey: process.env.PRIVATE_SECURITY_KEY
});

interface paymentParams {
    'OrderId': number,
    'Amount': string,
    'Currency': string,
    'ReturnUrl': string,
    'FailUrl': string
}

export const getUrl = async (params: paymentParams): Promise<string | undefined> => {
    return new Promise(function(resolve, reject) {
        payOnlineClient.getPaymentUrl(params, (err: string, url: string) => {
            if (err != undefined) {
                return reject(err);
            }
            console.log('Payment url: ' + url);
            return resolve(url);
        });  
    });
};

export const confirmPayment = async (callbackUrl: string): Promise<boolean> => {
    return new Promise(function(resolve) {
        payOnlineClient.parseCallback(callbackUrl, (err: string, result: any) => {
            if (err != undefined) {
                console.log(err);
                return resolve(false);
            }
            console.log('Success! Transaction ID: ' + result.TransactionID);
            return resolve(true);
        });   
    });
};