import {Request, Response} from 'express';
import {getConnection, getCustomRepository} from 'typeorm';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {PayonlineTransactions} from '../infrastructure/entity/PayonlineTransactions';
import {getUrl, confirmPayment} from '../services/payonlineService';
import {BotsRepository} from '../infrastructure/repositories/botsRepo';
import {TgNewDiagnosticRequestRepository} from '../infrastructure/repositories/tgNewDiagnosticRequestRepo';
import {BotUserRepository} from '../infrastructure/repositories/botUserRepo';
import {Bots as DomainBots} from '../domain/Bots';
import {diagnosticBot, localeService} from '../container';
import {InlineKeyboardButton} from 'node-telegram-bot-api';

type TransactionBody = {
    transaction_id: number,
    date_time: Date,
    order_id: number,
    is_confirmed: boolean
}

export class PaymentController {
    public async getTransaction (req: Request, res: Response) {
        let requestBody: TransactionBody;
        try {
            requestBody = {
                transaction_id: Number(req.query.TransactionID),
                date_time: new Date(String(req.query.DateTime)),
                order_id: Number(req.query.OrderId),
                is_confirmed: req.query.confirmed == 'true'
            };
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const connection = getConnection();
        let transaction: PayonlineTransactions;
        try {
            transaction = await connection
                .manager
                .findOneOrFail(PayonlineTransactions, {where: {id: requestBody.order_id}});
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        transaction.transaction_id = requestBody.transaction_id;
        transaction.date_updated = requestBody.date_time;
        transaction.is_confirmed = requestBody.is_confirmed;

        const splitUrl = req.originalUrl.split('?');
        const callbackParams = splitUrl[1];
        const isConfirmed = await confirmPayment(callbackParams);
        if (!isConfirmed) {
            transaction.is_confirmed = false;
        } 
        
        try {
            const transactionRepo = connection.getRepository(PayonlineTransactions);
            await transactionRepo.save(transaction);
            const firstBotId = await getCustomRepository(BotsRepository).
                findByStringOrFail(DomainBots.TG_COUGH_ANALYSIS);
            if (transaction.bot_id == firstBotId) { // there will be more ifs for more payment sources
                const chatId = await getCustomRepository(TgNewDiagnosticRequestRepository)
                    .findChatById(transaction.request_id);
                const botUser = await getCustomRepository(BotUserRepository)
                    .findOneByChatId(chatId);
                const lang = botUser.report_language;     
                if (transaction.is_confirmed) {
                    const inline_keyboard: InlineKeyboardButton[][] = [[{
                        text: localeService.translate({phrase: 'Next', locale: lang}),
                        callback_data: JSON.stringify({
                            'paid': 'true'
                        })
                    }]];
                    diagnosticBot.sendMessage(chatId, localeService.translate({phrase: 'Successful payment', locale: lang}), 
                        {reply_markup: {inline_keyboard: inline_keyboard}});
                } else {
                    // TODO: refactor button creation
                    const newTransaction = new PayonlineTransactions();
                    newTransaction.request_id = transaction.request_id;
                    newTransaction.bot_id = transaction.bot_id;
                    newTransaction.amount = transaction.amount;
                    newTransaction.currency = 'RUB';
                    await transactionRepo.save(newTransaction);

                    const orderId = newTransaction.id;
                    const formattedSum = (+newTransaction.amount).toFixed(2);
                    const paymentParams = {
                        'OrderId': orderId,
                        'Amount': formattedSum,
                        'Currency': 'RUB',
                        'ReturnUrl': 'https://t.me/cough_analysis_bot', 
                        'FailUrl': 'https://t.me/cough_analysis_bot'
                    };
                    const paymentUrl = await getUrl(paymentParams);

                    const inline_keyboard: InlineKeyboardButton[][] = [[{
                        text: `${localeService.translate({phrase: 'Pay roubles', locale: lang})}${newTransaction.amount} \uD83D\uDCB3`,
                        url: paymentUrl
                    }]];
                    const message = `${localeService.translate({phrase: 'Unsuccessful payment', locale: lang})} <a href="https://www.aicoughbot.com/contacts.html">${localeService.translate({phrase: 'contact support', locale: lang})}</a>.`;
                    diagnosticBot.sendMessage(chatId, message, {reply_markup: {inline_keyboard: inline_keyboard}, parse_mode: 'HTML', disable_web_page_preview: true});
                }
            }
            return res.status(HttpStatusCodes.NO_CONTENT).send();
        } catch(error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
    }
}