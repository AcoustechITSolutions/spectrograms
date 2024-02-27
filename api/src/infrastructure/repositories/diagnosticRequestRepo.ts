import {EntityRepository, Repository} from 'typeorm';
import {DiagnosticRequest} from '../entity/DiagnosticRequest';

@EntityRepository(DiagnosticRequest)
export class DiagnosticRequestRepository extends Repository<DiagnosticRequest> {
    public async findRequestByIdOrFail(requestId: number): Promise<DiagnosticRequest> {
        return this.findOneOrFail({where: {id: requestId}});
    }

    public async findRequestByIdAndUserIdOrFail(requestId: number, userId: number): Promise<DiagnosticRequest> {
        return this.findOneOrFail({where: {id: requestId, user_id: userId}});
    }
}
