import {EntityRepository, Repository, getConnection,} from 'typeorm';
import {NotifyDiagnosticBotUsers} from '../entity/NotifyDiagnosticBotUsers';

@EntityRepository(NotifyDiagnosticBotUsers)
export class NotifyDiagnosticBotUserRepository extends Repository<NotifyDiagnosticBotUsers> {
    public async findOneByChatId(chatId: number): Promise<NotifyDiagnosticBotUsers | undefined> {
        const connection = getConnection();
        const res = await connection.manager.findOne(NotifyDiagnosticBotUsers, {where: {chat_id: chatId}});
        return res;
    }
    
    public async findAllChatId(): Promise<number[]> {
        const notifyDiagnosticBotUsers = await this.createQueryBuilder('notify_diagnostic_bot_users')
            .select('notify_diagnostic_bot_users.chat_id')
            .getMany();
        return notifyDiagnosticBotUsers.map((notifyDiagnosticBotUsers) => notifyDiagnosticBotUsers.chat_id);
    }
}