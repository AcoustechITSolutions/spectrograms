import {getCustomRepository} from 'typeorm';
import {NotifyDiagnosticBotUserRepository} from '../../infrastructure/repositories/notifyDiagnosticBotUserRepo';
import {DoctorNotificationService} from './DoctorNotificationService';
import TelegramBot from 'node-telegram-bot-api';
import {localeService} from '../../container';

export class DoctorNotificationTelegramImpl extends DoctorNotificationService {

    constructor(private bot: TelegramBot) {
        super();
    }

    public async notifyAboutNewDiagnostic() {
        try {
            const notifyDiagnosticBotUserRepo = getCustomRepository(NotifyDiagnosticBotUserRepository);
            const notifyDiagnosticBotUsers = await notifyDiagnosticBotUserRepo.find();
            for (const botUser of notifyDiagnosticBotUsers) {
                try {
                    const publishResult = await this.bot.sendMessage(botUser.chat_id, localeService.translate({
                        phrase: this.NEW_DIAGNOSTIC_MESSAGE,
                        locale: botUser.language
                    }));
                    console.log(`Telegram message id: ${publishResult.message_id}`);
                } catch(error) {
                    const isChatIdForbiddenMessage: boolean = error.response && (error.response.statusCode === 403 || error.response.statusCode === 400);
                    if (isChatIdForbiddenMessage) {
                        await notifyDiagnosticBotUserRepo.remove(botUser);
                    }
                }
            }
        } catch(error) {
            console.error(error);
        }
    }

    public async notifyAboutSupportRequest(contactData: string, userMessage: string, userId?: number) {
        try {
            const notifyDiagnosticBotUserRepo = getCustomRepository(NotifyDiagnosticBotUserRepository);
            const notifyDiagnosticBotUsers = await notifyDiagnosticBotUserRepo.find();
            for (const botUser of notifyDiagnosticBotUsers) {
                try {
                    const title = localeService.translate({
                        phrase: 'support_request',
                        locale: botUser.language
                    });
                    const idText = localeService.translate({
                        phrase: 'user_id',
                        locale: botUser.language
                    });
                    const contactText = localeService.translate({
                        phrase: 'contact_data',
                        locale: botUser.language
                    });
                    const messageText = localeService.translate({
                        phrase: 'message',
                        locale: botUser.language
                    });
                    let msg: string;
                    if (userMessage == 'data_discharge') {
                        msg = localeService.translate({
                            phrase: 'data_discharge',
                            locale: botUser.language
                        });
                    } else {
                        msg = userMessage;
                    }
                    const supportMessage = `<b>${title}</b>\n\n${idText}: ${userId ?? '-'}\n${contactText}: ${contactData}\n${messageText}: ${msg}`;
                    const publishResult = await this.bot.sendMessage(botUser.chat_id, supportMessage, {parse_mode: 'HTML'});
                    console.log(`Telegram message id: ${publishResult.message_id}`);
                } catch(error) {
                    const isChatIdForbiddenMessage: boolean = error.response && (error.response.statusCode === 403 || error.response.statusCode === 400);
                    if (isChatIdForbiddenMessage) {
                        await notifyDiagnosticBotUserRepo.remove(botUser);
                    }
                }
            }
        } catch(error) {
            console.error(error);
        }
    }
}
