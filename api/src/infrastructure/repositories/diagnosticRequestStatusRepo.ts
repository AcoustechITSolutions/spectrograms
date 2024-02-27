import {EntityRepository, Repository} from 'typeorm';
import {DiagnosticRequestStatus} from '../entity/DiagnostRequestStatus';

@EntityRepository(DiagnosticRequestStatus)
export class DiagnosticRequestStatusRepository extends Repository<DiagnosticRequestStatus> {
    public async findByStringOrFail(status: string): Promise<DiagnosticRequestStatus> {
        return this.findOneOrFail({where: {request_status: status}});
    }

    public async findByStatusIdOrFail(statusId: number): Promise<DiagnosticRequestStatus> {
        return this.findOneOrFail({id: statusId});
    }
}
