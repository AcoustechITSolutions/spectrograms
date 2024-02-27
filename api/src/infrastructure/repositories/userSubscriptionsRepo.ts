import {EntityManager, EntityRepository, Repository} from 'typeorm';
import {UserSubscriptions} from '../entity/UserSubscriptions';

@EntityRepository(UserSubscriptions)
export class UserSubscriptionsRepository extends Repository<UserSubscriptions> {
    public async findByUserId(userId: number): Promise<UserSubscriptions> {
        return this.findOne({
            where: {user_id: userId}
        });
    }

    public async findByIdLocked(
        id: number, 
        manager: EntityManager
    ): Promise<UserSubscriptions> {
        return await manager
            .createQueryBuilder(UserSubscriptions, 'subs')
            .setLock('pessimistic_write')
            .where('subs.id = :sub_id', {sub_id: id})
            .getOne();
    }
}
