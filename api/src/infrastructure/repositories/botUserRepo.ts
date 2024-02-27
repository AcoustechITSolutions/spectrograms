import {EntityRepository, Repository, getConnection,} from 'typeorm';
import {BotUsers} from '../entity/BotUsers';

@EntityRepository(BotUsers)
export class BotUserRepository extends Repository<BotUsers> {
    public async findOneByChatId(chatId: number): Promise<BotUsers | undefined> {
        const connection = getConnection();
        const res = await connection.manager.findOne(BotUsers, {where: {chat_id: chatId}});
        return res;
    }
}