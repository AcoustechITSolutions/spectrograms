import {EntityRepository,  Repository} from 'typeorm';
import {TelegramDiagnosticRequestStatus as EntityTgStatus} from '../entity/TelegramDiagnosticRequestStatus';

@EntityRepository(EntityTgStatus)
export class TelegramDiagnosticRequestStatusRepository extends Repository<EntityTgStatus> {
    public async findByStringOrFail(status: string): Promise<EntityTgStatus> {
        return this.findOneOrFail({where: {request_status: status}});
    }

}
