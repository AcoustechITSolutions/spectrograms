import {Request, Response} from 'express';
import {InlineKeyboardButton} from 'node-telegram-bot-api';
import {HttpStatusCodes} from '../helpers/status';
import {muusBot} from '../container';
import {Readable} from 'stream';

import config from '../config/config';
import TelegramBot from 'node-telegram-bot-api';
import {getCustomRepository} from 'typeorm';
import {MuusDiagnosticRequestRepository} from '../infrastructure/repositories/muusDiagnosticRequestRepo';
import {TelegramDiagnosticRequestStatusRepository} from '../infrastructure/repositories/telegramDiagnosticRequestStatusRepo';
import {MuusDiagnosticRequest} from '../infrastructure/entity/MuusDiagnosticRequest';
import {DiagnosticRequestStatus, TelegramDiagnosticRequestStatus as DomainTgStatus} from '../domain/RequestStatus';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';
import {BotUserRepository} from '../infrastructure/repositories/botUserRepo';
import {BotUsers} from '../infrastructure/entity/BotUsers';
import {fileService} from '../container';

import {UserRepository} from '../infrastructure/repositories/userRepo';

import {inferenceDiagnostic} from '../services/diagnostic/diagnosticInferenceService';
import {PaginationParams} from '../interfaces/PaginationParams';
import {fetchDiagnosticByTelegramChatId} from '../services/query/diagnosticQueryService';
import {UserReportsResponse} from './diagnosticController';
import fs from 'fs';
import {join} from 'path';
import {createDiagnosticResultMessage} from '../services/diagnostic/muusBotNotificationService';
import {Gender} from '../domain/Gender';
import {localeService} from '../container';
import {persistAsDiagnosticRequest} from '../services/query/diagnosticBotQueryService';
import {isCoughPresented} from '../services/coughDetector';

const PAGINATION_LIMIT = 5;

muusBot.on('callback_query', async (cbQuery) => {
    const chatId = cbQuery.message.chat.id;
    return muusBot.sendMessage(chatId, 'Этот бот предназначался для тестирования участников молодежного фестиваля Muus uSTAR, организованного Университетом «Синергия». Работа бота остановлена. Подробнее о технологии Acoustery вы можете узнать на сайте www.acoustery.tech');
    
    const botUserRepo = getCustomRepository(BotUserRepository);
    let botUser = await botUserRepo.findOneByChatId(chatId);
    if (botUser == undefined) {
        const newUser = new BotUsers();
        newUser.chat_id = chatId;
        newUser.report_language = cbQuery.from.language_code ?? 'ru';
        await botUserRepo.save(newUser);
        botUser = newUser;
    }
    const lang = botUser.report_language;

    const requestRepo = getCustomRepository(MuusDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    const data = JSON.parse(cbQuery.data);
    if (data?.offset != undefined) {
        return onResult(cbQuery.message, data.offset);
    }

    if (req == undefined) {
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }
    
    switch(req.status.request_status) {
    case DomainTgStatus.GENDER: {
        const data = JSON.parse(cbQuery.data);
        if (data.gender == undefined)
            return sendQuestion(chatId, DomainTgStatus.GENDER, lang);
        const gender = await getCustomRepository(GenderTypesRepository)
            .findByStringOrFail(data.gender);
        req.gender = gender;
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgStatus.GENDER, cbQuery, lang);
         
        return toSmokingState(cbQuery.message, req);
    }
    case DomainTgStatus.IS_SMOKING: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_smoking == undefined)
            return sendQuestion(chatId, DomainTgStatus.IS_SMOKING, lang);
        req.is_smoking = JSON.parse(data.is_smoking);
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgStatus.IS_SMOKING, cbQuery, lang);

        return toCoughAudioState(cbQuery.message, req);
    }
    case DomainTgStatus.IS_FORCED: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_forced == undefined)
            return sendQuestion(chatId, DomainTgStatus.IS_FORCED, lang);
        req.is_forced = JSON.parse(data.is_forced);
        await requestRepo.save(req);

        await editReplyMarkup(DomainTgStatus.IS_FORCED, cbQuery, lang);
        return await onDone(cbQuery.message, req);
    }
    default: return sendQuestion(chatId, req.status.request_status, lang);
    }
});

muusBot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    return muusBot.sendMessage(chatId, 'Этот бот предназначался для тестирования участников молодежного фестиваля Muus uSTAR, организованного Университетом «Синергия». Работа бота остановлена. Подробнее о технологии Acoustery вы можете узнать на сайте www.acoustery.tech');
    const botUserRepo = getCustomRepository(BotUserRepository);
    let botUser = await botUserRepo.findOneByChatId(chatId);
    if (botUser == undefined) {
        const newUser = new BotUsers();
        newUser.chat_id = chatId;
        newUser.report_language = msg.from.language_code ?? 'ru';
        await botUserRepo.save(newUser);
        botUser = newUser;
    }
    const lang = botUser.report_language;

    if (msg.text == '/start') {
        return onStart(msg);
    }
    if (msg.text == '/result') {
        return onResult(msg);
    }
    
    const requestRepo = getCustomRepository(MuusDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    switch(req.status.request_status) {
    case DomainTgStatus.AGE: {
        const age = Number(msg.text);
        if (isNaN(age))
            return sendQuestion(chatId, DomainTgStatus.AGE, lang);
        if (age < 18 || age > 100) {
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'Acceptable age is between 18 and 100',
                locale: lang
            }));
        }
        req.age = age;
        await requestRepo.save(req);

        return toGenderState(msg, req);
    }
    case DomainTgStatus.COUGH_AUDIO: {
        if (msg?.voice == undefined)
            return sendQuestion(chatId, DomainTgStatus.COUGH_AUDIO, lang);
        if (msg?.voice?.duration < 8) {
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'The record duration should be at least 8 seconds.',
                locale: lang
            }));
        }

        let file: Readable;
        try {
            file = muusBot.getFileStream(msg.voice.file_id);
        } catch(error) {
            console.error(error);
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'Cannot get your audio file, try later.',
                locale: lang
            }));
        }
        const isCough = await isCoughPresented(file, 'ogg');
        if (isCough == undefined) {
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occured during the file processing, try later',
                locale: lang
            }));
        }
        if (!isCough) {
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'Cough is not detected in your record. Please, make a new record according to all the recommendations.',
                locale: lang
            }));
        }
        
        try {
            const audio_path = `${config.muusBotAudioFolder}/${req.id}/cough.ogg`;
            const fileStream = muusBot.getFileStream(msg.voice.file_id);
            const chunks = [];
            for await (const chunk of fileStream) {
                chunks.push(chunk);
            }
            const buffer = Buffer.concat(chunks);
            req.cough_audio_path = await fileService.saveFile(audio_path, buffer);
            await requestRepo.save(req);
        } catch(error) {
            console.error(error);
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occured during the file processing, try later.',
                locale: lang
            }));
        }

        return toForcedState(msg, req);
    }
    default: return sendQuestion(chatId, req.status.request_status, lang);
    }
});

muusBot.on('edited_message', async (msg) => {
    const chatId = msg.chat.id;
    return muusBot.sendMessage(chatId, 'Этот бот предназначался для тестирования участников молодежного фестиваля Muus uSTAR, организованного Университетом «Синергия». Работа бота остановлена. Подробнее о технологии Acoustery вы можете узнать на сайте www.acoustery.tech');
    const botUserRepo = getCustomRepository(BotUserRepository);
    const botUser = await botUserRepo.findOneByChatId(chatId);
    const lang = botUser.report_language;

    const requestRepo = getCustomRepository(MuusDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    if (req.age != undefined) { //you can edit age anytime, but not too early
        const age = Number(msg.text);
        if (isNaN(age))
            return sendQuestion(chatId, DomainTgStatus.AGE, lang);
        if (age < 18 || age > 100) {
            return muusBot.sendMessage(chatId, localeService.translate({
                phrase: 'Acceptable age is between 18 and 100',
                locale: lang
            }));
        }
        req.age = age;
        await requestRepo.save(req);
    } else {
        return sendQuestion(chatId, req.status.request_status, lang);
    }
});

export const editReplyMarkup = async (status: string, cbQuery: TelegramBot.CallbackQuery, lang: string) => {
    const data = JSON.parse(cbQuery.data);
    switch(status) {
    case DomainTgStatus.GENDER: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: data.gender == Gender.MALE ? 
                    `✓${localeService.translate({phrase: Gender.MALE, locale: lang})}` 
                    : localeService.translate({phrase: Gender.MALE, locale: lang}),
                callback_data: JSON.stringify({
                    'gender': 'male'
                })
            }, {
                text: data.gender == Gender.FEMALE ? 
                    `✓${localeService.translate({phrase: Gender.FEMALE, locale: lang})}`
                    : localeService.translate({phrase: Gender.FEMALE, locale: lang}),
                callback_data: JSON.stringify({
                    'gender': 'female'
                })
            }
        ]];
        return muusBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgStatus.IS_FORCED: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: Boolean(JSON.parse(data.is_forced)) ? 
                    `✓${localeService.translate({phrase: 'yes', locale: lang})}` 
                    : localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_forced': 'true'
                })
            }, {
                text: !Boolean(JSON.parse(data.is_forced)) ? 
                    `✓${localeService.translate({phrase: 'no', locale: lang})}`
                    : localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_forced': 'false'
                })
            }
        ]];
        return muusBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgStatus.IS_SMOKING: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: Boolean(JSON.parse(data.is_smoking)) ?   
                    `✓${localeService.translate({phrase: 'yes', locale: lang})}` 
                    : localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_smoking': 'true'
                })
            }, {
                text: !Boolean(JSON.parse(data.is_smoking)) ?  
                    `✓${localeService.translate({phrase: 'no', locale: lang})}`
                    : localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_smoking': 'false'
                })
            }
        ]];
        return muusBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    }
};

export const onDiagnosticBotMessage = async (req: Request, res: Response) => {
    muusBot.processUpdate(req.body);
    res.status(HttpStatusCodes.SUCCESS).send();
};

const onStart = async (msg: TelegramBot.Message) => {
    const chatId = msg.chat.id;
    const botUserRepo = getCustomRepository(BotUserRepository);
    let botUser = await botUserRepo.findOneByChatId(chatId);
    if (botUser == undefined) {
        const newUser = new BotUsers();
        newUser.chat_id = chatId;
        newUser.report_language = msg.from.language_code ?? 'ru';
        await botUserRepo.save(newUser);
        botUser = newUser;
    }
    const lang = botUser.report_language; 

    const requestRepo = getCustomRepository(MuusDiagnosticRequestRepository);
    const ids = await requestRepo.findNonCancelledRequestsIds(chatId);
    if (ids.length > 0) {
        await requestRepo.cancelRequestByIds(ids);
    }
    const statusRepo = getCustomRepository(TelegramDiagnosticRequestStatusRepository);
    const ageStatus = await statusRepo.findByStringOrFail(DomainTgStatus.AGE);
    const newRequest = new MuusDiagnosticRequest();
    newRequest.status = ageStatus;
    newRequest.chat_id = chatId;
    await requestRepo.save(newRequest);

    return sendQuestion(chatId, DomainTgStatus.AGE, lang);
};

const sendQuestion = async (chatId: number, state: string, lang: string, req?: MuusDiagnosticRequest) => {
    switch (state) {
    case DomainTgStatus.AGE:  return muusBot.sendMessage(chatId, localeService.translate({
        phrase: 'How old are you?',
        locale: lang
    }));
    case DomainTgStatus.GENDER: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: Gender.MALE, locale: lang}),
                callback_data: JSON.stringify({
                    'gender': 'male'
                })
            }, {
                text: localeService.translate({phrase: Gender.FEMALE, locale: lang}),
                callback_data: JSON.stringify({
                    'gender': 'female'
                })
            }]
        ];
            
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Indicate your gender',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgStatus.IS_FORCED: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_forced': 'true'
                })
            }, {
                text: localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_forced': 'false'
                })
            }]
        ];
        
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Indicate if the cough was forced',
            locale: lang
        }),
        {reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgStatus.IS_SMOKING: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_smoking': 'true'
                })
            }, {
                text: localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_smoking': 'false'
                })
            }]
        ];
        
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'Do you smoke?',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgStatus.COUGH_AUDIO: {
        const photoStream = fs.createReadStream(join(__dirname, '../../static/tgbot/muus_cough.png'));
        const message = `${localeService.translate({phrase: 'diagnostic_bot_send_voice', locale: lang})}
<b>${localeService.translate({phrase: 'avoid_noise', locale: lang})}</b>`;
        return muusBot.sendPhoto(chatId, photoStream, {caption: message, parse_mode: 'HTML'});
    }
    case DomainTgStatus.DONE: {
        const repo = getCustomRepository(MuusDiagnosticRequestRepository);
        const date = await repo.findDateByDiagnosticRequestId(req.request_id);
        const message = `${localeService.translate({phrase: 'success_message', locale: lang})} ${date}.`;
        return muusBot.sendMessage(chatId, message);
    }
}
};

const toGenderState = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const genderState = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.GENDER);
    req.status = genderState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(MuusDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.GENDER, botUser.report_language);
};

const toForcedState = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const forcedState = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.IS_FORCED);
    req.status =  forcedState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(MuusDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.IS_FORCED, botUser.report_language);  
};

const toSmokingState = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const smokingState = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.IS_SMOKING);
    req.status = smokingState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(MuusDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.IS_SMOKING, botUser.report_language);
};

const toCoughAudioState = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const coughAudioState = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.COUGH_AUDIO);
    req.status = coughAudioState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(MuusDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.COUGH_AUDIO, botUser.report_language);
};

const toDoneState = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const doneState = await getCustomRepository(TelegramDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.DONE);
    req.status = doneState;
    req.date_finished = new Date();
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(MuusDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.DONE, botUser.report_language, req);
};

const onDone = async (msg: TelegramBot.Message, req: MuusDiagnosticRequest) => {
    const userRepo = getCustomRepository(UserRepository);
    const userId = await userRepo.findMuusBotUserId();
    const reqId = await persistAsDiagnosticRequest(req, userId);
    await toDoneState(msg, req);
    await inferenceDiagnostic(req.cough_audio_path, reqId);
};

const onResult = async (msg: TelegramBot.Message, offset?: number) => {
    const paginationParams: PaginationParams = {
        limit: PAGINATION_LIMIT + 1,
        offset: offset ?? 0
    };
    const chatId = msg.chat.id;
    const lang = msg.from.language_code;
    
    const userId = await getCustomRepository(UserRepository)
        .findMuusBotUserId();
    const userReports = await fetchDiagnosticByTelegramChatId(chatId, paginationParams, userId);

    const response: UserReportsResponse[] = userReports.map((userReport) => {
        const status = userReport.status;
        const date = userReport.date;
        const request_id = userReport.request_id;
        const report = status == DiagnosticRequestStatus.SUCCESS ? {
            probability: userReport.probability ?? undefined,
            diagnosis: userReport.diagnosis ?? undefined,
            productivity: userReport.productivity ?? undefined,
            intensity: userReport.intensity ?? undefined,
            commentary: userReport.commentary ?? undefined,
        } : undefined;
        return {status, date, request_id, report};
    });

    if (response.length == 0)
        return muusBot.sendMessage(chatId, localeService.translate({
            phrase: 'You have not taken any tests yet',
            locale: lang
        }));

    let message = `${localeService.translate({phrase: 'Diagnostic results', locale: lang})}:\n`;
    for (const i in response) {
        if (Number(i) == PAGINATION_LIMIT) {
            continue;
        }
        const data = response[i];
        const repo = getCustomRepository(MuusDiagnosticRequestRepository);
        const date = await repo.findDateByDiagnosticRequestId(data.request_id);
        const recordNum = Number(i) + 1 + paginationParams.offset;
        if (data.status == DiagnosticRequestStatus.SUCCESS) {
            message = message.concat(`\n<b>${recordNum}. ${date}</b>\n${createDiagnosticResultMessage(data.report, lang)}\n`);
        } else {
            message = message.concat(`\n<b>${recordNum}. ${date}</b>\n${localeService.translate({phrase: data.status, locale: lang})}\n`);
        }
    }
    const inlineKeyboard: InlineKeyboardButton[][] = [[]];
    if (paginationParams.offset > 0) {
        inlineKeyboard[0].push({
            text: 'Назад',
            callback_data: JSON.stringify({
                'offset': paginationParams.offset - PAGINATION_LIMIT
            })
        });
    }
    if (response.length > PAGINATION_LIMIT) {
        inlineKeyboard[0].push({
            text: 'Дальше',
            callback_data: JSON.stringify({
                'offset': paginationParams.offset + PAGINATION_LIMIT
            })
        });
    }
   
    return muusBot.sendMessage(chatId, message, {
        parse_mode: 'HTML', reply_markup: {inline_keyboard: inlineKeyboard}
    });
};
