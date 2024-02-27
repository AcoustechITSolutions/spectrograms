import {S3, SNS, SES, Pinpoint} from 'aws-sdk';
import TelegramBot from 'node-telegram-bot-api';
import config from './config/config';
import {LocaleService} from './services/locales/localeService';
import {join} from 'path';

import SftpClient from 'ssh2-sftp-client';
import {SftpFileAccessServiceImpl} from './services/file/SftpFileAccessServiceImpl';
import {S3FileAccessServiceImpl} from './services/file/S3FileAccessServiceImpl';
import {FileAccessService} from './services/file/FileAccessService';
import {DoctorNotificationService} from './services/notification/DoctorNotificationService';
import {DoctorNotificationServiceSmsImpl} from './services/notification/DoctorNotificationSmsImpl';
import {DoctorNotificationTelegramImpl} from './services/notification/DoctorNotificationTelegramImpl';
import {DoctorNotificationEmptyImpl} from './services/notification/DoctorNotificationEmptyImpl';

import {SendUserSms} from './services/notification/userSmsService';
import {SendUserEmail} from './services/notification/userEmailService';

const token = config.tgBotToken;
const url = config.tgBotPublicUrl;
const dataToken = config.datasetBotToken;
const dataUrl = config.datasetBotPublicUrl;
const diagnosticToken = config.diagnosticBotToken;
const diagnosticUrl = config.diagnosticBotPublicUrl;
const muusToken = config.muusBotToken;
const muusUrl = config.muusBotPublicUrl;
const notifyDiagnosticToken = config.notifyDiagnosticBotToken;
const notifyDiagnosticUrl = config.notifyDiagnosticBotPublicUrl;
const regWebhook = config.registerWebhook;

// TODO: replace all of this by di containers

const awsStandartParams = {
    accessKeyId: process.env.S3_USER_ID,
    secretAccessKey: process.env.S3_USER_SECRET,
    region: 'eu-central-1',
    endpoint: process.env.NODE_ENV != 'production' ? process.env.LOCALSTACK_ENDPOINT : undefined
};
const awsNotificationParams = { 
    accessKeyId: process.env.AWS_USER_ID, 
    secretAccessKey: process.env.AWS_USER_SECRET,
    region: 'eu-central-1'
};
const pinpoint = new Pinpoint(awsNotificationParams);
const notificationSns = new SNS(awsNotificationParams);
const notificationSes = new SES(awsNotificationParams);
const notificationSmsService = new SendUserSms(notificationSns);
const notificationEmailService = new SendUserEmail(notificationSes);

const notifyDiagnosticBot = new TelegramBot(notifyDiagnosticToken, {
    webHook: {
        port: 8446, 
    }
});
if (regWebhook) {
    notifyDiagnosticBot.setWebHook(`${notifyDiagnosticUrl}/bot${notifyDiagnosticToken}`);
}

let doctorNotificationService: DoctorNotificationService;
if (config.doctorNotificationType == 'sms') {
    doctorNotificationService = new DoctorNotificationServiceSmsImpl(notificationSns);
} else if (config.doctorNotificationType == 'telegram') {
    doctorNotificationService = new DoctorNotificationTelegramImpl(notifyDiagnosticBot);
} else {
    doctorNotificationService = new DoctorNotificationEmptyImpl();
}

const localeService = new LocaleService({
    locales: ['en', 'ru'],
    defaultLocale: 'en',
    updateFiles: false,
    directory: join(__dirname, 'infrastructure/locales'),
});

let fileService: FileAccessService;
if (config.fileAccessProtocol == 's3') {
    const s3 = new S3({
        ...awsStandartParams,
        s3ForcePathStyle: process.env.NODE_ENV != 'production' ? true : undefined,
    });
    fileService = new S3FileAccessServiceImpl(s3, config.s3Bucket, config.datasetSpectreFolder);
}
    
else if (config.fileAccessProtocol == 'sftp') {
    const sftp = new SftpClient();
    sftp.connect({
        host: config.sftpHost,
        port: Number(config.sftpPort),
        username: config.sftpUsername,
        password: config.sftpPassword,
        retries: 3
    });
    fileService = new SftpFileAccessServiceImpl(sftp, config.sftpUploadFolder, config.datasetSpectreFolder);
}
    
else
    throw Error('no file access protocol provided.');

const bot = new TelegramBot(token, {
    webHook: true
});
if (regWebhook) {
    bot.setWebHook(`${url}/bot${token}`);
}

const dataBot = new TelegramBot(dataToken, {
    webHook: {
        port: 8444,  
    }
});
if (regWebhook) {
    dataBot.setWebHook(`${dataUrl}/bot${dataToken}`);
}

const diagnosticBot = new TelegramBot(diagnosticToken, {
    webHook: {
        port: 8445, 
    }
});
if (regWebhook) {
    diagnosticBot.setWebHook(`${diagnosticUrl}/bot${diagnosticToken}`);
}

const muusBot = new TelegramBot(muusToken, {
    webHook: {
        port: 8447, 
    }
});
if (regWebhook) {
    muusBot.setWebHook(`${muusUrl}/bot${muusToken}`);
}

export {
    bot,
    dataBot,
    diagnosticBot,
    muusBot,
    notifyDiagnosticBot,
    token,
    dataToken,
    diagnosticToken,
    muusToken,
    notifyDiagnosticToken,
    localeService,
    fileService,
    doctorNotificationService,
    notificationSns,
    notificationSmsService,
    notificationEmailService,
    pinpoint
};
