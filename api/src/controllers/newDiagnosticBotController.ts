import {Request, Response} from 'express';
import {InlineKeyboardButton} from 'node-telegram-bot-api';
import {HttpStatusCodes} from '../helpers/status';
import {diagnosticBot} from '../container';
import {Readable} from 'stream';

import config from '../config/config';
import TelegramBot from 'node-telegram-bot-api';
import {getCustomRepository, getConnection} from 'typeorm';
import {TgNewDiagnosticRequestRepository} from '../infrastructure/repositories/tgNewDiagnosticRequestRepo';
import {TgNewDiagnosticRequestStatusRepository} from '../infrastructure/repositories/tgNewDiagnosticRequestStatusRepo';
import {TgNewDiagnosticRequest} from '../infrastructure/entity/TgNewDiagnosticRequest';
import {BotPayments} from '../infrastructure/entity/BotPayments';
import {BotUsers} from '../infrastructure/entity/BotUsers';
import {PayonlineTransactions} from '../infrastructure/entity/PayonlineTransactions';
import {TgNewDiagnosticRequestStatus as DomainTgStatus} from '../domain/RequestStatus';
import {GenderTypesRepository} from '../infrastructure/repositories/genderRepo';
import {BotsRepository} from '../infrastructure/repositories/botsRepo';
import {BotUserRepository} from '../infrastructure/repositories/botUserRepo';
import {BotPaymentRepository} from '../infrastructure/repositories/botPaymentRepo';
import {PaymentTypesRepository} from '../infrastructure/repositories/paymentTypesRepo';
import {fileService} from '../container';
import {UserRepository, DIAGNOSTIC_BOT_USER} from '../infrastructure/repositories/userRepo';

import {inferenceDiagnostic} from '../services/diagnostic/diagnosticInferenceService';
import fs from 'fs';
import {join} from 'path';
import {Gender} from '../domain/Gender';
import {Payments} from '../domain/Payments';
import {Bots as DomainBots} from '../domain/Bots';
import {localeService} from '../container';
import {persistAsDiagnosticRequest} from '../services/query/diagnosticBotQueryService';
import {getUrl} from '../services/payonlineService';
import {coughValidation} from '../services/coughValidation';

export const singlePayment = 30; 
export const unlimitedPayment = 300;

diagnosticBot.on('callback_query', async (cbQuery) => {
    const chatId = cbQuery.message.chat.id;
    const user = await getCustomRepository(UserRepository).findByLogin(DIAGNOSTIC_BOT_USER);
    if (!user.is_active) {
        return diagnosticBot.sendMessage(chatId, 'The bot is not available at the moment');
    }
    const data = JSON.parse(cbQuery.data);
    if (data.new == 'true')
        return onStart(cbQuery.message);
    
    const botUserRepo = getCustomRepository(BotUserRepository);
    let botUser = await botUserRepo.findOneByChatId(chatId);
    if (botUser == undefined) {
        const newUser = new BotUsers();
        newUser.chat_id = chatId;
        newUser.report_language = data.language ?? cbQuery.from.language_code ?? 'ru';
        await botUserRepo.save(newUser);
        botUser = newUser;
    }
    const lang = botUser.report_language;

    const requestRepo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);
    if (req == undefined && data.language == undefined) {
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    if (data.language != undefined) {
        const customLang = data.language;
        botUser.report_language = customLang;
        await botUserRepo.save(botUser);
        await editReplyMarkup(DomainTgStatus.LANGUAGE, cbQuery, customLang);
        if (req == undefined) {
            return onStart(cbQuery.message);
        } else {
            return sendQuestion(chatId, req.status.request_status, customLang);
        }
    }

    const botId = await getCustomRepository(BotsRepository).
        findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
    switch(req.status.request_status) {
    case DomainTgStatus.START: {
        const data = JSON.parse(cbQuery.data);
        if (data.start == undefined) {
            return sendQuestion(chatId, DomainTgStatus.START, lang);
        }
        await editReplyMarkup(DomainTgStatus.START, cbQuery, lang);
        return toSupportState(cbQuery.message, req);
    }
    case DomainTgStatus.SUPPORT: {
        const data = JSON.parse(cbQuery.data);
        if (data.support == undefined) 
            return sendQuestion(chatId, DomainTgStatus.SUPPORT, lang);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
            const botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
            if (data.support == 'free') {
                botPayment.is_active = true;
                await botPaymentRepo.save(botPayment, {transaction: false});
                await queryRunner.commitTransaction();
                await editReplyMarkup(DomainTgStatus.SUPPORT, cbQuery, lang, req); //uses transaction
                return toDisclaimerState(cbQuery.message, req);
            } else if (data.support == 'pay_30') {
                req.payment_sum = singlePayment;
                botPayment.payment_type = await getCustomRepository(PaymentTypesRepository)
                    .findByStringOrFail(Payments.SINGLE);
            } else if (data.support == 'pay_300') {
                req.payment_sum = unlimitedPayment;
                botPayment.payment_type = await getCustomRepository(PaymentTypesRepository)
                    .findByStringOrFail(Payments.UNLIMITED);
            } else if (data.support == 'other') {
                botPayment.payment_type = await getCustomRepository(PaymentTypesRepository)
                    .findByStringOrFail(Payments.OTHER);
            }
            await requestRepo.save(req, {transaction: false});
            await botPaymentRepo.save(botPayment, {transaction: false});
            await queryRunner.commitTransaction();
            await editReplyMarkup(DomainTgStatus.SUPPORT, cbQuery, lang, req); //uses transaction
            return toPaymentState(cbQuery.message, req);
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }
    case DomainTgStatus.PAYMENT: {
        const data = JSON.parse(cbQuery.data);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
            const botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
            const other = await getCustomRepository(PaymentTypesRepository)
                .findByStringOrFail(Payments.OTHER);
            if (data.paid == undefined || botPayment.payment_type_id == other.id) {
                await queryRunner.rollbackTransaction();
                return sendQuestion(chatId, DomainTgStatus.PAYMENT, lang, req);
            }
            botPayment.is_active = true;
            await botPaymentRepo.save(botPayment, {transaction: false});
            await queryRunner.commitTransaction();
            await editReplyMarkup(DomainTgStatus.PAYMENT, cbQuery, lang, req); 
            return toDisclaimerState(cbQuery.message, req);
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }
    case DomainTgStatus.DISCLAIMER: {
        const data = JSON.parse(cbQuery.data);
        if (data.disclaimer == undefined)
            return sendQuestion(chatId, DomainTgStatus.DISCLAIMER, lang);
        await editReplyMarkup(DomainTgStatus.DISCLAIMER, cbQuery, lang);
        return toConditionsState(cbQuery.message, req);
    }
    case DomainTgStatus.CONDITIONS: {
        const data = JSON.parse(cbQuery.data);
        if (data.conditions == undefined)
            return sendQuestion(chatId, DomainTgStatus.CONDITIONS, lang);
        await editReplyMarkup(DomainTgStatus.CONDITIONS, cbQuery, lang);
        return toAgeState(cbQuery.message, req);
    }
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
    default: return sendQuestion(chatId, req.status.request_status, lang, req);
    }
});

diagnosticBot.on('message', async (msg) => {
    const chatId = msg.chat.id;
    const user = await getCustomRepository(UserRepository).findByLogin(DIAGNOSTIC_BOT_USER);
    if (!user.is_active) {
        return diagnosticBot.sendMessage(chatId, 'The bot is not available at the moment');
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

    if (msg.text == '/start') {
        return onStart(msg);
    }

    if (msg.text == '/language') {
        return sendQuestion(chatId, DomainTgStatus.LANGUAGE, lang);
    }

    const requestRepo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    const botId = await getCustomRepository(BotsRepository).
        findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
    switch(req.status.request_status) {
    case DomainTgStatus.PAYMENT: {
        const paymentSum = Number(msg.text);
        if (isNaN(paymentSum)) 
            return sendQuestion(chatId, DomainTgStatus.PAYMENT, lang, req);   
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
            const paymentTypesRepo = getCustomRepository(PaymentTypesRepository);
            const botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
            const free = await paymentTypesRepo.findByStringOrFail(Payments.FREE);
            const single = await paymentTypesRepo.findByStringOrFail(Payments.SINGLE);
            const unlimited = await paymentTypesRepo.findByStringOrFail(Payments.UNLIMITED);
            if (botPayment.payment_type_id == single.id || botPayment.payment_type_id == unlimited.id) {
                // if the user has already chosen the sum but still sends a message
                await queryRunner.rollbackTransaction();
                return sendQuestion(chatId, DomainTgStatus.PAYMENT, lang, req);
            }  

            if (paymentSum < singlePayment && botPayment.payment_type_id != free.id) {
                await queryRunner.rollbackTransaction();
                return diagnosticBot.sendMessage(chatId, localeService.translate({
                    phrase: 'Acceptable payment is more than 30 roubles',
                    locale: lang
                }));
            }

            if (paymentSum >= unlimitedPayment) {
                botPayment.payment_type = unlimited;
            } else {
                botPayment.payment_type = single;
            }
            req.payment_sum = paymentSum;
            await requestRepo.save(req, {transaction: false});
            await botPaymentRepo.save(botPayment, {transaction: false});
            await queryRunner.commitTransaction();
            return toPaymentState(msg, req);
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }
    case DomainTgStatus.AGE: {
        const age = Number(msg.text);
        if (isNaN(age))
            return sendQuestion(chatId, DomainTgStatus.AGE, lang);
        if (age < 18 || age > 100) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
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
        if (msg?.voice?.duration < 3) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'The record duration should be at least 3 seconds.',
                locale: lang
            }));
        }

        let file: Readable;
        try {
            file = diagnosticBot.getFileStream(msg.voice.file_id);
        } catch(error) {
            console.error(error);
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'Cannot get your audio file, try later.',
                locale: lang
            }));
        }
        const validationResponse = await coughValidation(file, 'ogg');
        if (validationResponse == undefined) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occurred during the file processing, try later',
                locale: lang
            }));
        }
        const isCough = validationResponse.is_cough;
        const isEnough = validationResponse.is_enough; 
        const isClear = validationResponse.is_clear;
        if (!isCough) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'Cough is not detected in your record. Please, make a new record according to all the recommendations.',
                locale: lang
            }));
        }
        const user = await getCustomRepository(UserRepository).findByLogin(DIAGNOSTIC_BOT_USER);
        if (!isEnough && user.is_validate_cough) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'not_enough_cough_message',
                locale: lang
            }));
        }
        if (!isClear) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'noisy_cough_message',
                locale: lang
            }));
        }
        
        try {
            const audio_path = `${config.diagnosticBotAudioFolder}/${req.id}/cough.ogg`;
            const fileStream = diagnosticBot.getFileStream(msg.voice.file_id);
            const chunks = [];
            for await (const chunk of fileStream) {
                chunks.push(chunk);
            }
            const buffer = Buffer.concat(chunks);
            req.cough_audio_path = await fileService.saveFile(audio_path, buffer);
            await requestRepo.save(req);
        } catch(error) {
            console.error(error);
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'An error occured during the file processing, try later.',
                locale: lang
            }));
        }

        return await onDone(msg, req);
    }
    default: return sendQuestion(chatId, req.status.request_status, lang, req);
    }
});

diagnosticBot.on('edited_message', async (msg) => {
    const chatId = msg.chat.id;
    const user = await getCustomRepository(UserRepository).findByLogin(DIAGNOSTIC_BOT_USER);
    if (!user.is_active) {
        return diagnosticBot.sendMessage(chatId, 'The bot is not available at the moment');
    }
    const botUserRepo = getCustomRepository(BotUserRepository);
    const botUser = await botUserRepo.findOneByChatId(chatId);
    const lang = botUser.report_language;

    const requestRepo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const req = await requestRepo.findNonCancelledRequest(chatId);

    if (req == undefined) {
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'Unsupported command, you can start working with the bot using the /start command.',
            locale: lang
        }));
    }

    if (req.age != undefined) { //you can edit age anytime, but not too early, not to mess up the payment
        const age = Number(msg.text);
        if (isNaN(age))
            return sendQuestion(chatId, DomainTgStatus.AGE, lang);
        if (age < 18 || age > 100) {
            return diagnosticBot.sendMessage(chatId, localeService.translate({
                phrase: 'Acceptable age is between 18 and 100',
                locale: lang
            }));
        }
        req.age = age;
        await requestRepo.save(req);
    } else {
        return sendQuestion(chatId, req.status.request_status, lang, req);
    }
});

export const editReplyMarkup = async (status: string, cbQuery: TelegramBot.CallbackQuery, lang: string, req?: TgNewDiagnosticRequest) => {
    const data = JSON.parse(cbQuery.data);
    switch(status) {
    case DomainTgStatus.LANGUAGE: {
        const markup: InlineKeyboardButton[][] = [[
            {
                text: data.language == 'en' ? 
                    `✓${localeService.translate({phrase: 'en', locale: lang})}` 
                    : localeService.translate({phrase: 'en', locale: lang}),
                callback_data: JSON.stringify({
                    'language': 'en'
                })
            }], [{
            text: data.language == 'ru' ? 
                `✓${localeService.translate({phrase: 'ru', locale: lang})}`
                : localeService.translate({phrase: 'ru', locale: lang}),
            callback_data: JSON.stringify({
                'language': 'ru'
            })
        }]];
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }

    case DomainTgStatus.START: {
        const markup: InlineKeyboardButton[][] = [[
            { 
                text: data.start == 'true' ? 
                    `✓${localeService.translate({phrase: 'Support the project', locale: lang})}` 
                    : localeService.translate({phrase: 'Support the project', locale: lang}),
                callback_data: JSON.stringify({
                    'start': 'true'
                })
            }
        ]];
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }

    case DomainTgStatus.SUPPORT: {
        const botId = await getCustomRepository(BotsRepository)
            .findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
        const free = await getCustomRepository(PaymentTypesRepository)
            .findByStringOrFail(Payments.FREE);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPayment = await manager.getCustomRepository(BotPaymentRepository)
                .findOneByChatId(req.chat_id, botId, manager);
            let markup: InlineKeyboardButton[][];
            if (botPayment.payment_type_id == free.id) {
                markup = [[{
                    text: data.support == 'free' ? 
                        `✓${localeService.translate({phrase: 'free', locale: lang})}` 
                        : localeService.translate({phrase: 'free', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'free'
                    })
                }], [{
                    text: data.support == 'pay_30' ? 
                        `✓${localeService.translate({phrase: 'pay_30', locale: lang})}` 
                        : localeService.translate({phrase: 'pay_30', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'pay_30'
                    })
                }], [{
                    text: data.support == 'pay_300' ? 
                        `✓${localeService.translate({phrase: 'pay_300', locale: lang})}` 
                        : localeService.translate({phrase: 'pay_300', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'pay_300'
                    })
                }], [{
                    text: data.support == 'other' ? 
                        `✓${localeService.translate({phrase: 'other', locale: lang})}` 
                        : localeService.translate({phrase: 'other', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'other'
                    })
                }]];
            } else {
                markup = [[{
                    text: data.support == 'pay_30' ? 
                        `✓${localeService.translate({phrase: 'pay_30', locale: lang})}` 
                        : localeService.translate({phrase: 'pay_30', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'pay_30'
                    })
                }], [{
                    text: data.support == 'pay_300' ? 
                        `✓${localeService.translate({phrase: 'pay_300', locale: lang})}` 
                        : localeService.translate({phrase: 'pay_300', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'pay_300'
                    })
                }], [{
                    text: data.support == 'other' ? 
                        `✓${localeService.translate({phrase: 'other', locale: lang})}` 
                        : localeService.translate({phrase: 'other', locale: lang}),
                    callback_data: JSON.stringify({
                        'support': 'other'
                    })
                }]];
            }
            await queryRunner.commitTransaction();
            return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
                message_id: cbQuery.message.message_id,
                chat_id: cbQuery.message.chat.id,
                inline_message_id: cbQuery.inline_message_id
            });
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }

    case DomainTgStatus.PAYMENT: {
        const markup: InlineKeyboardButton[][] = [[{ 
            text: data.paid == 'true' ? 
                `✓${localeService.translate({phrase: 'Next', locale: lang})}`  
                : localeService.translate({phrase: 'Next', locale: lang}),
            callback_data: JSON.stringify({
                'paid': 'true'
            })
        }
        ]];
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }

    case DomainTgStatus.DISCLAIMER: {
        const markup: InlineKeyboardButton[][] = [[
            { 
                text: data.disclaimer == 'true' ? 
                    `✓${localeService.translate({phrase: 'Agree', locale: lang})}` 
                    : localeService.translate({phrase: 'Agree', locale: lang}),
                callback_data: JSON.stringify({
                    'disclaimer': 'true'
                })
            }
        ]];
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }

    case DomainTgStatus.CONDITIONS: {
        const markup: InlineKeyboardButton[][] = [[
            { 
                text: data.conditions == 'true' ? 
                    `✓${localeService.translate({phrase: 'Agree', locale: lang})}` 
                    : localeService.translate({phrase: 'Agree', locale: lang}),
                callback_data: JSON.stringify({
                    'conditions': 'true'
                })
            }
        ]];
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }

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
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
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
        return diagnosticBot.editMessageReplyMarkup({inline_keyboard: markup}, {
            message_id: cbQuery.message.message_id,
            chat_id: cbQuery.message.chat.id,
            inline_message_id: cbQuery.inline_message_id
        });
    }
    }
};

export const onDiagnosticBotMessage = async (req: Request, res: Response) => {
    diagnosticBot.processUpdate(req.body);
    res.status(HttpStatusCodes.SUCCESS).send();
};

export const onStart = async (msg: TelegramBot.Message) => {
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

    const newRequest = new TgNewDiagnosticRequest();
    newRequest.chat_id = chatId;
    const statusRepo = getCustomRepository(TgNewDiagnosticRequestStatusRepository);
    const startStatus = await statusRepo.findByStringOrFail(DomainTgStatus.START);
    const supportStatus = await statusRepo.findByStringOrFail(DomainTgStatus.SUPPORT);
    const disclaimerStatus = await statusRepo.findByStringOrFail(DomainTgStatus.DISCLAIMER);
    let status: string;

    const botId = await getCustomRepository(BotsRepository).
        findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
    const connection = getConnection();
    const queryRunner = connection.createQueryRunner();
    await queryRunner.startTransaction();
    const manager = queryRunner.manager;
    try {
        const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
        let botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
        if (botPayment == undefined) {
            const firstPayment = new BotPayments();
            firstPayment.chat_id = chatId;
            firstPayment.bot_id = botId;
            const free = await getCustomRepository(PaymentTypesRepository)
                .findByStringOrFail(Payments.FREE);
            firstPayment.payment_type = free;
            await botPaymentRepo.save(firstPayment, {transaction: false});
            botPayment = firstPayment;
            newRequest.status = startStatus;
            status = DomainTgStatus.START;
        } else {
            if (botPayment.is_active) {
                newRequest.status = disclaimerStatus;
                status = DomainTgStatus.DISCLAIMER;
            } else {
                newRequest.status = supportStatus;
                status = DomainTgStatus.SUPPORT;
            }
        }
        await queryRunner.commitTransaction();
    } catch (err) {
        await queryRunner.rollbackTransaction();
    } finally {
        await queryRunner.release();
    }

    const requestRepo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const ids = await requestRepo.findNonCancelledRequestsIds(chatId);
    if (ids.length > 0) {
        await requestRepo.cancelRequestByIds(ids);
    }
    await requestRepo.save(newRequest);
    return sendQuestion(chatId, status, lang);
};

const sendQuestion = async (chatId: number, state: string, lang: string, req?: TgNewDiagnosticRequest) => {
    switch (state) {
    case DomainTgStatus.LANGUAGE: {
        const inline_keyboard: InlineKeyboardButton[][] = [
            [{
                text: localeService.translate({phrase: 'en', locale: lang}),
                callback_data: JSON.stringify({
                    'language': 'en'
                })
            }], [{
                text: localeService.translate({phrase: 'ru', locale: lang}),
                callback_data: JSON.stringify({
                    'language': 'ru'
                })
            }]
        ];
                
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'choose_language',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }

    case DomainTgStatus.START: {
        const inline_keyboard: InlineKeyboardButton[][] = [[{
            text: localeService.translate({phrase: 'Support the project', locale: lang}),
            callback_data: JSON.stringify({
                'start': 'true'
            })
        }]];
        const start_message = `<b>${localeService.translate({phrase: 'start_header', locale: lang})}</b>\n${localeService.translate({phrase: 'start_message', locale: lang})}`;
        return diagnosticBot.sendMessage(chatId, start_message, {
            parse_mode: 'HTML', 
            reply_markup: {inline_keyboard: inline_keyboard}});
    }

    case DomainTgStatus.SUPPORT: {
        let inline_keyboard: InlineKeyboardButton[][];
        let supportMessage: string;
        const botId = await getCustomRepository(BotsRepository)
            .findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
        const free = await getCustomRepository(PaymentTypesRepository)
            .findByStringOrFail(Payments.FREE);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPayment = await manager.getCustomRepository(BotPaymentRepository)
                .findOneByChatId(chatId, botId, manager);    
            if (botPayment.payment_type_id == free.id) {
                supportMessage = localeService.translate({phrase: 'support_message_first', locale: lang});
                inline_keyboard = [
                    [{
                        text: localeService.translate({phrase: 'free', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'free'
                        })
                    }], [{
                        text: localeService.translate({phrase: 'pay_30', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'pay_30'
                        })
                    }], [{
                        text: localeService.translate({phrase: 'pay_300', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'pay_300'
                        })
                    }], [{
                        text: localeService.translate({phrase: 'other', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'other'
                        }) 
                    }]];
            } else {
                supportMessage = localeService.translate({phrase: 'support_message_second', locale: lang});
                inline_keyboard = [
                    [{
                        text: localeService.translate({phrase: 'pay_30', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'pay_30'
                        })
                    }], [{
                        text: localeService.translate({phrase: 'pay_300', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'pay_300'
                        })
                    }], [{
                        text: localeService.translate({phrase: 'other', locale: lang}),
                        callback_data: JSON.stringify({
                            'support': 'other'
                        }) 
                    }]];
            }
            
            await queryRunner.commitTransaction();
            return diagnosticBot.sendMessage(chatId, supportMessage, {
                reply_markup: {inline_keyboard: inline_keyboard}});    
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }

    case DomainTgStatus.PAYMENT: {
        const botId = await getCustomRepository(BotsRepository)
            .findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
        const other = await getCustomRepository(PaymentTypesRepository)
            .findByStringOrFail(Payments.OTHER);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPayment = await manager.getCustomRepository(BotPaymentRepository)
                .findOneByChatId(chatId, botId, manager);
            if (botPayment.payment_type_id == other.id) {
                await queryRunner.rollbackTransaction();
                return diagnosticBot.sendMessage(chatId, localeService.translate({phrase: 'Enter your sum', locale: lang}));
            } else {
                // TODO: refactor button creation
                const transaction = new PayonlineTransactions();
                transaction.request_id = req.id;
                transaction.bot_id = botId;
                transaction.amount = req.payment_sum;
                transaction.currency = 'RUB';
                const connection = getConnection();
                const transactionRepo = connection.getRepository(PayonlineTransactions);
                await transactionRepo.save(transaction);

                const orderId = transaction.id;
                const formattedSum = (+req.payment_sum).toFixed(2);
                const paymentParams = {
                    'OrderId': orderId,
                    'Amount': formattedSum,
                    'Currency': 'RUB',
                    'ReturnUrl': 'https://t.me/cough_analysis_bot', 
                    'FailUrl': 'https://t.me/cough_analysis_bot'
                };
                const paymentUrl = await getUrl(paymentParams);

                const inline_keyboard: InlineKeyboardButton[][] = [[{
                    text: `${localeService.translate({phrase: 'Pay roubles', locale: lang})}${req.payment_sum} \uD83D\uDCB3`,
                    url: paymentUrl
                }]];

                await queryRunner.commitTransaction();
                return diagnosticBot.sendMessage(chatId, localeService.translate({phrase: 'for_payment', locale: lang}), {
                    parse_mode: 'HTML',
                    reply_markup: {inline_keyboard: inline_keyboard}});
            }  
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }

    case DomainTgStatus.DISCLAIMER: {
        const inline_keyboard: InlineKeyboardButton[][] = [[{
            text: localeService.translate({phrase: 'Agree', locale: lang}),
            callback_data: JSON.stringify({
                'disclaimer': 'true'
            })
        }]];

        const disclaimer_message = `<b>${localeService.translate({phrase: 'disclaimer_header', locale: lang})}</b>\n${localeService.translate({phrase: 'disclaimer_message', locale: lang})}`;
        return diagnosticBot.sendMessage(chatId, disclaimer_message, {
            parse_mode: 'HTML', 
            reply_markup: {inline_keyboard: inline_keyboard}});
    }

    case DomainTgStatus.CONDITIONS: {
        const inline_keyboard: InlineKeyboardButton[][] = [[{
            text: localeService.translate({phrase: 'Agree', locale: lang}),
            callback_data: JSON.stringify({
                'conditions': 'true'
            })
        }]];

        const conditions_message = `<b>${localeService.translate({phrase: 'conditions_header', locale: lang})}</b>
<a href="http://www.aicoughbot.com/index.html#rule">${localeService.translate({phrase: 'Read conditions', locale: lang})}</a>`;

        return diagnosticBot.sendMessage(chatId, conditions_message, {
            parse_mode: 'HTML', 
            disable_web_page_preview: true,
            reply_markup: {inline_keyboard: inline_keyboard}});
    }

    case DomainTgStatus.AGE:  return diagnosticBot.sendMessage(chatId, localeService.translate({
        phrase: 'age_message',
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
            
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'Indicate your gender',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
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
        
        return diagnosticBot.sendMessage(chatId, localeService.translate({
            phrase: 'smoke_message',
            locale: lang
        }), {
            reply_markup: {inline_keyboard: inline_keyboard}});
    }

    case DomainTgStatus.COUGH_AUDIO: {
        const photoStream = fs.createReadStream(join(__dirname, '../../static/tgbot/10cm.png'));
        return diagnosticBot.sendPhoto(chatId, photoStream,
            {caption: localeService.translate({
                phrase: 'diagnostic_bot_send_voice',
                locale: lang
            })});
    }

    case DomainTgStatus.DONE: {
        const repo = getCustomRepository(TgNewDiagnosticRequestRepository);
        const date = await repo.findDateByDiagnosticRequestId(req.request_id);
        const message = `${localeService.translate({phrase: 'success_message', locale: lang})} <b>${date}</b>`;
        return diagnosticBot.sendMessage(chatId, message, {parse_mode: 'HTML'});
    } 
    }
};

const toSupportState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const supportState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.SUPPORT);
    req.status = supportState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.SUPPORT, botUser.report_language);
};

const toPaymentState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const paymentState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.PAYMENT);
    req.status = paymentState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.PAYMENT, botUser.report_language, req);
};

const toDisclaimerState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const disclaimerState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.DISCLAIMER);
    req.status = disclaimerState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.DISCLAIMER, botUser.report_language);
};

const toConditionsState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const conditionsState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.CONDITIONS);
    req.status = conditionsState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.CONDITIONS, botUser.report_language);
};

const toAgeState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const ageState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.AGE);
    req.status = ageState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.AGE, botUser.report_language);
};

const toGenderState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const genderState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.GENDER);
    req.status = genderState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.GENDER, botUser.report_language);
};

const toSmokingState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const smokingState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.IS_SMOKING);
    req.status = smokingState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.IS_SMOKING, botUser.report_language);
};

const toCoughAudioState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const coughAudioState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.COUGH_AUDIO);
    req.status = coughAudioState;
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.COUGH_AUDIO, botUser.report_language);
};

const toDoneState = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const doneState = await getCustomRepository(TgNewDiagnosticRequestStatusRepository)
        .findByStringOrFail(DomainTgStatus.DONE);
    req.status = doneState;
    req.date_finished = new Date();
    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(req.chat_id);
    await getCustomRepository(TgNewDiagnosticRequestRepository).save(req);
    return sendQuestion(msg.chat.id, DomainTgStatus.DONE, botUser.report_language, req);
};

const onDone = async (msg: TelegramBot.Message, req: TgNewDiagnosticRequest) => {
    const botId = await getCustomRepository(BotsRepository).
        findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
    const connection = getConnection();
    const queryRunner = connection.createQueryRunner();
    await queryRunner.startTransaction();
    const manager = queryRunner.manager;
    try {
        const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
        const paymentTypesRepo = getCustomRepository(PaymentTypesRepository);
        const botPayment = await botPaymentRepo.findOneByChatId(req.chat_id, botId, manager);
        const unlimited = await paymentTypesRepo.findByStringOrFail(Payments.UNLIMITED);
        const single = await paymentTypesRepo.findByStringOrFail(Payments.SINGLE);
        if (botPayment.payment_type_id != unlimited.id) {
            botPayment.is_active = false;
            botPayment.payment_type = single;
            await botPaymentRepo.save(botPayment, {transaction: false});
        }
        await queryRunner.commitTransaction();
    } catch (err) {
        await queryRunner.rollbackTransaction();
    } finally {
        await queryRunner.release();
    }
    const userRepo = getCustomRepository(UserRepository);
    const userId = await userRepo.findDiagnosticBotUserId();
    const reqId = await persistAsDiagnosticRequest(req, userId);
    await toDoneState(msg, req);
    await inferenceDiagnostic(req.cough_audio_path, reqId);
};