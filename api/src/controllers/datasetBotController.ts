import {Request, Response} from 'express';
import {InlineKeyboardButton} from 'node-telegram-bot-api';
import {HttpStatusCodes} from '../helpers/status';
import {dataBot} from '../container';
import {Readable} from 'stream';

import config from '../config/config';
import TelegramBot from 'node-telegram-bot-api';
import {getCustomRepository} from 'typeorm';
import {TelegramDatasetRequestRepository} from '../infrastructure/repositories/telegramDatasetRequestRepo';
import {TelegramDatasetRequestStatusRepository} from '../infrastructure/repositories/telegramDatasetRequestStatusRepo';
import {TelegramDatasetRequest} from '../infrastructure/entity/TelegramDatasetRequest';
import {TelegramDatasetRequestStatus as DomainTgDataStatus} from '../domain/RequestStatus';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';
import {BotUserRepository} from '../infrastructure/repositories/botUserRepo';
import {BotUsers} from '../infrastructure/entity/BotUsers';
import {fileService} from '../container';

import {UserRepository, DATASET_BOT_USER} from '../infrastructure/repositories/userRepo';

import {sendDataset} from '../services/sendDatasetService';
import {persistAsDatasetRequest} from '../services/query/datasetBotQueryService';
import fs from 'fs';
import {join} from 'path';
import {Gender} from '../domain/Gender';
import {localeService} from '../container';
import {coughValidation} from '../services/coughValidation';

dataBot.on('callback_query', async (cbQuery) => {
    const chatId = cbQuery.message.chat.id;
    const user = await getCustomRepository(UserRepository).findByLogin(DATASET_BOT_USER);
    if (!user.is_active) {
        return dataBot.sendMessage(chatId, 'The bot is not available at the moment');
    }
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

    const requestRepo = getCustomRepository(TelegramDatasetRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    switch(req.status.request_status) {
    case DomainTgDataStatus.GENDER: {
        const data = JSON.parse(cbQuery.data);
        if (data.gender == undefined)                       //if clicked on a button from another question
            return sendQuestion(chatId, DomainTgDataStatus.GENDER, lang);
        const gender = await getCustomRepository(GenderTypesRepository)
            .findByStringOrFail(data.gender);
        req.gender = gender;
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgDataStatus.GENDER, cbQuery, lang);

        return toSmokingState(cbQuery.message, req);
    }
    case DomainTgDataStatus.IS_SMOKING: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_smoking == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.IS_SMOKING, lang);
        req.is_smoking = JSON.parse(data.is_smoking);
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgDataStatus.IS_SMOKING, cbQuery, lang);

        return toCovidState(cbQuery.message, req);
    }
    case DomainTgDataStatus.IS_COVID: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_covid == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.IS_COVID, lang);
        req.is_covid = JSON.parse(data.is_covid);
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgDataStatus.IS_COVID, cbQuery, lang);

        return toDiseaseState(cbQuery.message, req);
    }
    case DomainTgDataStatus.IS_DISEASE: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_disease == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.IS_DISEASE, lang);
        req.is_disease = JSON.parse(data.is_disease);
        await requestRepo.save(req);
        await editReplyMarkup(DomainTgDataStatus.IS_DISEASE, cbQuery, lang);
        if (req.is_disease) {
            return toDiseaseNameState(cbQuery.message, req);
        } else {
            req.disease_name = null;
            return toCoughAudioState(cbQuery.message, req);
        }
    }
    case DomainTgDataStatus.IS_FORCED: {
        const data = JSON.parse(cbQuery.data);
        if (data.is_forced == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.IS_FORCED, lang);
        req.is_forced = JSON.parse(data.is_forced);
        await requestRepo.save(req);

        await editReplyMarkup(DomainTgDataStatus.IS_FORCED, cbQuery, lang);
        await toDoneState(cbQuery.message, req);
        return onDone(req);
    }
    default: return sendQuestion(chatId, req.status.request_status, lang);
    }
});

dataBot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    const user = await getCustomRepository(UserRepository).findByLogin(DATASET_BOT_USER);
    if (!user.is_active) {
        return dataBot.sendMessage(chatId, 'The bot is not available at the moment');
    }
    if (msg.text == '/start') {
        return onStart(msg);
    }

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

    const requestRepo = getCustomRepository(TelegramDatasetRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    switch(req.status.request_status) {
    case DomainTgDataStatus.AGE: {
        const age = Number(msg.text);
        if (isNaN(age))
            return sendQuestion(chatId, DomainTgDataStatus.AGE, lang);
        if (age < 18 || age > 100) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'Acceptable age is between 18 and 100',
                locale: lang
            }));
        }
        req.age = age;
        await requestRepo.save(req);

        return toGenderState(msg, req);
    }
    case DomainTgDataStatus.DISEASE_NAME: {
        const disease_name = msg.text;
        if (disease_name == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.DISEASE_NAME, lang);
        req.disease_name = disease_name;
        await requestRepo.save(req);

        return toCoughAudioState(msg, req);
    }
    case DomainTgDataStatus.COUGH_AUDIO: {
        if (msg?.voice == undefined)
            return sendQuestion(chatId, DomainTgDataStatus.COUGH_AUDIO, lang);
        if (msg?.voice?.duration < 3) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'The record duration should be at least 3 seconds.',
                locale: lang
            }));
        }

        let file: Readable;
        try {
            file = dataBot.getFileStream(msg.voice.file_id);
        } catch(error) {
            console.error(error);
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'Impossible to get your audio file, try later.',
                locale: lang
            }));
        }
        const validationResponse = await coughValidation(file, 'ogg');
        if (validationResponse == undefined) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occurred during the file processing, try later',
                locale: lang
            }));
        }
        const isCough = validationResponse.is_cough;
        const isEnough = validationResponse.is_enough; 
        const isClear = validationResponse.is_clear;
        if (!isCough) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'Cough is not detected in your record. Please, make a new record according to all the recommendations.',
                locale: lang
            }));
        }
        const user = await getCustomRepository(UserRepository).findByLogin(DATASET_BOT_USER);
        if (!isEnough && user.is_validate_cough) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'not_enough_cough_message',
                locale: lang
            }));
        }
        if (!isClear) {
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'noisy_cough_message',
                locale: lang
            }));
        }

        try {
            const audio_path = `${config.datasetBotAudioFolder}/${req.id}/cough.ogg`;
            const fileStream = dataBot.getFileStream(msg.voice.file_id);
            const chunks = [];
            for await (const chunk of fileStream) {
                chunks.push(chunk);
            }
            const buffer = Buffer.concat(chunks);
            req.cough_audio_path = await fileService.saveFile(audio_path, buffer);
            await requestRepo.save(req);
        } catch(error) {
            console.error(error);
            return dataBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occurred during the file processing, try later.',
                locale: lang
            }));
        }

        return toForcedState(msg, req);
    }
    default: return sendQuestion(chatId, req.status.request_status, lang);
    }
});

export const editReplyMarkup = async (status: string, cbQuery: TelegramBot.CallbackQuery, lang: string) => {
    const data = JSON.parse(cbQuery.data);
    switch(status) {
    case DomainTgDataStatus.GENDER: {
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
        return dataBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgDataStatus.IS_SMOKING: {
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
        return dataBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgDataStatus.IS_COVID: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: Boolean(JSON.parse(data.is_covid)) ?
                    `✓${localeService.translate({phrase: 'yes', locale: lang})}`
                    : localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_covid': 'true'
                })
            }, {
                text: !Boolean(JSON.parse(data.is_covid)) ?
                    `✓${localeService.translate({phrase: 'no', locale: lang})}`
                    : localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_covid': 'false'
                })
            }
        ]];
        return dataBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgDataStatus.IS_DISEASE: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: Boolean(JSON.parse(data.is_disease)) ?
                    `✓${localeService.translate({phrase: 'yes', locale: lang})}`
                    : localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_disease': 'true'
                })
            }, {
                text: !Boolean(JSON.parse(data.is_disease)) ?
                    `✓${localeService.translate({phrase: 'no', locale: lang})}`
                    : localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_disease': 'false'
                })
            }
        ]];
        return dataBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    case DomainTgDataStatus.IS_FORCED: {
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
        return dataBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    }
};

export const onDatasetBotMessage = async (req: Request, res: Response) => {
    dataBot.processUpdate(req.body);
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
    
    const requestRepo = getCustomRepository(TelegramDatasetRequestRepository);
    const ids = await requestRepo.findNonCancelledRequestsIds(chatId);
    if (ids.length > 0) {
        await requestRepo.cancelRequestByIds(ids);
    }
    const statusRepo = getCustomRepository(TelegramDatasetRequestStatusRepository);
    const ageStatus = await statusRepo.findByStringOrFail(DomainTgDataStatus.AGE);
    const newRequest = new TelegramDatasetRequest();
    newRequest.status = ageStatus;
    newRequest.chat_id = chatId;
    await requestRepo.save(newRequest);

    return sendQuestion(chatId, DomainTgDataStatus.AGE, lang);
};

const sendQuestion = async (chatId: number, state: string, lang: string) => {
    switch (state) {
    case DomainTgDataStatus.AGE:  return dataBot.sendMessage(chatId, localeService.translate({
        phrase: 'How old are you?',
        locale: lang
    }));
    case DomainTgDataStatus.GENDER: {
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

        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Indicate your gender',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgDataStatus.IS_SMOKING: {
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

        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Do you smoke?',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgDataStatus.IS_COVID: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_covid': 'true'
                })
            }, {
                text: localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_covid': 'false'
                })
            }]
        ];

        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Do you have diagnosed COVID-19 at the moment?',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgDataStatus.IS_DISEASE: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: 'yes', locale: lang}),
                callback_data: JSON.stringify({
                    'is_disease': 'true'
                })
            }, {
                text: localeService.translate({phrase: 'no', locale: lang}),
                callback_data: JSON.stringify({
                    'is_disease': 'false'
                })
            }]
        ];

        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Do you have any other acute or chronic respiratory diseases except for COVID-19?',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgDataStatus.DISEASE_NAME:  return dataBot.sendMessage(chatId, localeService.translate({
        phrase: 'Indicate your diagnosis and/or symptoms',
        locale: lang
    }));
    case DomainTgDataStatus.IS_FORCED: {
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

        return dataBot.sendMessage(chatId, localeService.translate({
            phrase: 'Indicate if the cough was forced',
            locale: lang
        }),
        {reply_markup: {inline_keyboard: inline_keyboard}});
    }
    case DomainTgDataStatus.COUGH_AUDIO: {
        const photoStream = fs.createReadStream(join(__dirname, '../../static/tgbot/Cough.png'));
        return dataBot.sendPhoto(chatId, photoStream,
            {caption: localeService.translate({
                phrase: 'tg_bot_send_voice',
                locale: lang
            })});
    }
    case DomainTgDataStatus.DONE: return dataBot.sendMessage(chatId, localeService.translate({
        phrase: 'Done! We have received you data. Thank you for your collaboration!',
        locale: lang
    }));
    }
};

const toGenderState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const genderState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.GENDER);
    req.status = genderState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.GENDER, botUser.report_language);
};

const toSmokingState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const smokingState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.IS_SMOKING);
    req.status = smokingState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.IS_SMOKING, botUser.report_language);
};

const toCovidState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const covidState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.IS_COVID);
    req.status = covidState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.IS_COVID, botUser.report_language);
};

const toDiseaseState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const diseaseState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.IS_DISEASE);
    req.status = diseaseState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.IS_DISEASE, botUser.report_language);
};

const toDiseaseNameState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const diseaseNameState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.DISEASE_NAME);
    req.status = diseaseNameState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.DISEASE_NAME, botUser.report_language);
};

const toForcedState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const forcedState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.IS_FORCED);
    req.status =  forcedState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.IS_FORCED, botUser.report_language);
};

const toCoughAudioState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const coughAudioState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.COUGH_AUDIO);
    req.status = coughAudioState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.COUGH_AUDIO, botUser.report_language);
};

const toDoneState = async (msg: TelegramBot.Message, req: TelegramDatasetRequest) => {
    const doneState = await getCustomRepository(TelegramDatasetRequestStatusRepository)
        .findByStringOrFail(DomainTgDataStatus.DONE);
    req.status = doneState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TelegramDatasetRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgDataStatus.DONE, botUser.report_language);
};

const onDone = async (req: TelegramDatasetRequest) => {
    const userRepo = getCustomRepository(UserRepository);
    const userId = await userRepo.findDatasetBotUserId();
    const reqId = await persistAsDatasetRequest(req, userId);
    await sendDataset(reqId);
};
