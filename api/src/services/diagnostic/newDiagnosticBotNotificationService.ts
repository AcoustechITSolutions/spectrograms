import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {getCustomRepository, getConnection} from 'typeorm';
import {diagnosticBot} from '../../container';
import {InlineKeyboardButton} from 'node-telegram-bot-api';
import {fetchDiagnosticWithChatIdByRequestId} from '../query/diagnosticQueryService';

import {TgNewDiagnosticRequestRepository} from '../../infrastructure/repositories/tgNewDiagnosticRequestRepo';
import {UserDiagnosticReport} from '../../controllers/diagnosticController';
import {DiagnosisTypes} from '../../domain/DiagnosisTypes';
import {localeService} from '../../container';
import {UserRepository} from '../../infrastructure/repositories/userRepo';
import {BotsRepository} from '../../infrastructure/repositories/botsRepo';
import {BotUserRepository} from '../../infrastructure/repositories/botUserRepo';
import {BotPaymentRepository} from '../../infrastructure/repositories/botPaymentRepo';
import {PaymentTypesRepository} from '../../infrastructure/repositories/paymentTypesRepo';
import {Payments} from '../../domain/Payments';
import {Bots as DomainBots} from '../../domain/Bots';

export const onPatchNewDiagnostic = async (requestId: number) => {
    const userId = await getCustomRepository(UserRepository)
        .findDiagnosticBotUserId();
    const report = await fetchDiagnosticWithChatIdByRequestId(requestId, userId);

    if (report?.tg_new_chat_id == undefined)
        return;

    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(report.tg_new_chat_id);
    const lang = botUser.report_language ?? 'ru'; 
    const repo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const date = await repo.findDateByDiagnosticRequestId(requestId);
    let message = '';
    if (report.status == DiagnosticRequestStatus.SUCCESS) {
        message = message.concat(`<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'data processed', locale: lang})}\n   
${createDiagnosticResultMessage(report, lang)}`);
    } else {
        message = message.concat(`<b>${localeService.translate({phrase: 'Request status', locale: lang})} ${date}</b> 
${localeService.translate({phrase: report.status, locale: lang})}`);
    }
    const inline_keyboard: InlineKeyboardButton[][] = [[{
        text: localeService.translate({phrase: 'new test', locale: lang}),
        callback_data: JSON.stringify({
            'new': 'true'
        })
    }]];
    return diagnosticBot.sendMessage(report.tg_new_chat_id, message, {parse_mode: 'HTML', reply_markup: {inline_keyboard: inline_keyboard}});
};

export const onNewNoisyRecord = async (requestId: number) => {
    const repo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const userId = await getCustomRepository(UserRepository)
        .findDiagnosticBotUserId();
    const chatId = await repo.findChatIdByDiagnosticRequestId(requestId, userId);
    if (chatId != undefined) {
        const botId = await getCustomRepository(BotsRepository).
            findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
            const botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
            const unlimited = await getCustomRepository(PaymentTypesRepository)
                .findByStringOrFail(Payments.UNLIMITED);
            const botUser = await getCustomRepository(BotUserRepository)
                .findOneByChatId(chatId);
            const lang = botUser.report_language ?? 'ru'; 
            const date = await repo.findDateByDiagnosticRequestId(requestId);
            let message: string;
            if (botPayment.payment_type_id != unlimited.id) {
                botPayment.is_active = true;
                await botPaymentRepo.save(botPayment);
                message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'noisy_result', locale: lang})}`;
            } else {
                message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'noisy_audio', locale: lang})}`;
            }
            
            const inline_keyboard: InlineKeyboardButton[][] = [[{
                text: localeService.translate({phrase: 'new test', locale: lang}),
                callback_data: JSON.stringify({
                    'new': 'true'
                })
            }]];
            await queryRunner.commitTransaction();
            return diagnosticBot.sendMessage(chatId, message, {parse_mode: 'HTML', reply_markup: {inline_keyboard: inline_keyboard}}); 
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }
};

export const onNewDiagnosticError = async (requestId: number) => {
    const repo = getCustomRepository(TgNewDiagnosticRequestRepository);
    const userId = await getCustomRepository(UserRepository)
        .findDiagnosticBotUserId();
    const chatId = await repo.findChatIdByDiagnosticRequestId(requestId, userId);
    if (chatId != undefined) {
        const botId = await getCustomRepository(BotsRepository).
            findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            const botPaymentRepo = manager.getCustomRepository(BotPaymentRepository);
            const botPayment = await botPaymentRepo.findOneByChatId(chatId, botId, manager);
            const unlimited = await getCustomRepository(PaymentTypesRepository)
                .findByStringOrFail(Payments.UNLIMITED);
            const botUser = await getCustomRepository(BotUserRepository)
                .findOneByChatId(chatId);
            const lang = botUser.report_language ?? 'ru'; 
            const date = await repo.findDateByDiagnosticRequestId(requestId);
            let message: string;
            if (botPayment.payment_type_id != unlimited.id) {
                botPayment.is_active = true;
                await botPaymentRepo.save(botPayment);
                message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'error_result', locale: lang})}`;
            } else {
                message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'error', locale: lang})}`;
            }
        
            const inline_keyboard: InlineKeyboardButton[][] = [[{
                text: localeService.translate({phrase: 'new test', locale: lang}),
                callback_data: JSON.stringify({
                    'new': 'true'
                })
            }]];
            await queryRunner.commitTransaction();
            return diagnosticBot.sendMessage(chatId, message, {parse_mode: 'HTML', reply_markup: {inline_keyboard: inline_keyboard}});
        } catch (err) {
            await queryRunner.rollbackTransaction();
        } finally {
            await queryRunner.release();
        }
    }
};

export const createDiagnosticResultMessage = (report: UserDiagnosticReport, lang: string) => {
    let report_message = `${localeService.translate({phrase: 'Diagnosis', locale: lang})}: <b>${localeService.translate({phrase: report.diagnosis, locale: lang})}</b>`;
    if (report?.diagnosis == DiagnosisTypes.COVID_19) {
        report_message = report_message.concat(`\n${localeService.translate({phrase: 'recommendation', locale: lang})}`);
    }
    return report_message;
};
