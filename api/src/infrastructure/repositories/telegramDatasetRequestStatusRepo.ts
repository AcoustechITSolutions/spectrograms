import {EntityRepository,  Repository} from 'typeorm';
import {TelegramDatasetRequestStatus as EntityTgDataStatus} from '../entity/TelegramDatasetRequestStatus';

@EntityRepository(EntityTgDataStatus)
export class TelegramDatasetRequestStatusRepository extends Repository<EntityTgDataStatus> {
    public async findByStringOrFail(status: string): Promise<EntityTgDataStatus> {
        return this.findOneOrFail({where: {request_status: status}});
    }
}
