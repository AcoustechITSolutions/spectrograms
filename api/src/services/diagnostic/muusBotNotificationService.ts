import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {getCustomRepository} from 'typeorm';
import {muusBot} from '../../container';
import {fetchDiagnosticWithChatIdByRequestId} from '../query/diagnosticQueryService';

import {MuusDiagnosticRequestRepository} from '../../infrastructure/repositories/muusDiagnosticRequestRepo';
import {BotUserRepository} from '../../infrastructure/repositories/botUserRepo';
import {UserDiagnosticReport} from '../../controllers/diagnosticController';
import {DiagnosisTypes} from '../../domain/DiagnosisTypes';
import {localeService} from '../../container';
import {UserRepository} from '../../infrastructure/repositories/userRepo';

export const onPatchMuusDiagnostic = async (requestId: number) => {
    const userId = await getCustomRepository(UserRepository)
        .findMuusBotUserId();
    const report = await fetchDiagnosticWithChatIdByRequestId(requestId, userId);

    if (report?.muus_chat_id == undefined)
        return;

    const botUser = await getCustomRepository(BotUserRepository)
        .findOneByChatId(report.muus_chat_id);
    const lang = botUser.report_language ?? 'en'; 
    let message = '';
    const repo = getCustomRepository(MuusDiagnosticRequestRepository);
    const date = await repo.findDateByDiagnosticRequestId(requestId);
    if (report.status == DiagnosticRequestStatus.SUCCESS) 
        message = message.concat(`${localeService.translate({phrase: 'Diagnostic result', locale: lang})} ${date}\n
${createDiagnosticResultMessage(report, report.report_language)}`);
    else
        message = message.concat(`${localeService.translate({phrase: 'Request status', locale: lang})} ${date}: ${localeService.translate({phrase: report.status, locale: lang})}`);
    
    return muusBot.sendMessage(report.muus_chat_id, message, {parse_mode: 'HTML'});
};

export const onMuusNoisyRecord = async (requestId: number) => {
    const repo = getCustomRepository(MuusDiagnosticRequestRepository);
    const userId = await getCustomRepository(UserRepository)
        .findMuusBotUserId();
    const chatId = await repo.findChatIdByDiagnosticRequestId(requestId, userId);
    if (chatId != undefined) {
        const botUser = await getCustomRepository(BotUserRepository)
            .findOneByChatId(chatId);
        const lang = botUser.report_language ?? 'en'; 
        const date = await repo.findDateByDiagnosticRequestId(requestId);
        const message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'noisy_result_free', locale: lang})}`;
        return muusBot.sendMessage(chatId, message, {parse_mode: 'HTML'});
    }
};

export const onMuusDiagnosticError = async (requestId: number) => {
    const repo = getCustomRepository(MuusDiagnosticRequestRepository);
    const userId = await getCustomRepository(UserRepository)
        .findMuusBotUserId();
    const chatId = await repo.findChatIdByDiagnosticRequestId(requestId, userId);
    if (chatId != undefined) {
        const botUser = await getCustomRepository(BotUserRepository)
            .findOneByChatId(chatId);
        const lang = botUser.report_language ?? 'en';
        const date = await repo.findDateByDiagnosticRequestId(requestId);
        const message = `<b>${localeService.translate({phrase: 'result', locale: lang})} ${date}</b>
${localeService.translate({phrase: 'error_result_free', locale: lang})}`;
        return muusBot.sendMessage(chatId, message, {parse_mode: 'HTML'});
    }
};

export const createDiagnosticResultMessage = (report: UserDiagnosticReport, lang: string) => {
    let report_message = `${localeService.translate({phrase: 'Diagnosis', locale: lang})}: <b>${localeService.translate({phrase: report.diagnosis, locale: lang})}</b>`;
    if (report.diagnosis == DiagnosisTypes.COVID_19) {
        report_message = report_message.concat(`\n${localeService.translate({phrase: 'recommendation', locale: lang})}`);
    }
    return report_message;
};
