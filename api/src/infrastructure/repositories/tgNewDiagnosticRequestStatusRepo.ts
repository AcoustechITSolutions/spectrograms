import {EntityRepository,  Repository} from 'typeorm';
import {TgNewDiagnosticRequestStatus as EntityTgStatus} from '../entity/TgNewDiagnosticRequestStatus';

@EntityRepository(EntityTgStatus)
export class TgNewDiagnosticRequestStatusRepository extends Repository<EntityTgStatus> {
    public async findByStringOrFail(status: string): Promise<EntityTgStatus> {
        return this.findOneOrFail({where: {request_status: status}});
    }

}
