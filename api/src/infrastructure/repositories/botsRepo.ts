import {EntityRepository, getConnection, Repository} from 'typeorm';
import {Bots} from '../entity/Bots';

@EntityRepository(Bots)
export class BotsRepository extends Repository<Bots> {
    public async findByStringOrFail(name: string): Promise<number> {
        const connection = getConnection();
        const bot = await connection.manager.findOneOrFail(Bots, {where: {bot_name: name}});
        return bot.id;
    }
}