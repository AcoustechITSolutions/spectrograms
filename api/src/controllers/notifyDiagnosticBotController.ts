import {Request, Response} from 'express';
import {HttpStatusCodes} from '../helpers/status';
import {notifyDiagnosticBot} from '../container';
import TelegramBot from 'node-telegram-bot-api';
import {getCustomRepository} from 'typeorm';
import {localeService} from '../container';
import {NotifyDiagnosticBotUsers} from '../infrastructure/entity/NotifyDiagnosticBotUsers';
import {NotifyDiagnosticBotUserRepository} from '../infrastructure/repositories/notifyDiagnosticBotUserRepo';

notifyDiagnosticBot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    const botUserRepo = getCustomRepository(NotifyDiagnosticBotUserRepository);
    let botUser = await botUserRepo.findOneByChatId(chatId);
    if (botUser == undefined) {
        const newUser = new NotifyDiagnosticBotUsers();
        newUser.chat_id = chatId;
        newUser.language = msg.from.language_code ?? 'ru';
        await botUserRepo.save(newUser);
        botUser = newUser;
    }
    const lang = botUser.language;

    if (msg.text == '/start') {
        return onStart(msg, lang);
    }
    return notifyDiagnosticBot.sendMessage(chatId, localeService.translate({
        phrase: 'Unsupported command, you can start working with the bot using the /start command.',
        locale: lang
    }));
});

export const onNotifyDiagnosticBotMessage = async (req: Request, res: Response) => {
    notifyDiagnosticBot.processUpdate(req.body);
    res.status(HttpStatusCodes.SUCCESS).send();
};

const onStart = async (msg: TelegramBot.Message, lang: string) => {
    return notifyDiagnosticBot.sendMessage(msg.chat.id, localeService.translate({
        phrase: 'Notifications enabled.',
        locale: lang
    }));
};
