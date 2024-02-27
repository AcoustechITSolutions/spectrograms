import {EntityRepository, Repository,  EntityManager} from 'typeorm';
import {BotPayments} from '../entity/BotPayments';

@EntityRepository(BotPayments)
export class BotPaymentRepository extends Repository<BotPayments> {
    public async findOneByChatId(chatId: number, botId: number, manager: EntityManager): Promise<BotPayments | undefined> {
        const res = await manager
            .createQueryBuilder(BotPayments, 'payment')
            .setLock('pessimistic_write')
            .where('payment.chat_id = :chat_id', {chat_id: chatId})
            .andWhere('payment.bot_id = :bot_id', {bot_id: botId})
            .getOne(); 

        return res;
    }
}