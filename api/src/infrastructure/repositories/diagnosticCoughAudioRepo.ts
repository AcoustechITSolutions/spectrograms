import {EntityRepository, Repository} from 'typeorm';
import {CoughAudio} from '../entity/CoughAudio';

@EntityRepository(CoughAudio)
export class DiagnosticCoughAudioRepository extends Repository<CoughAudio> {
    public async findByRequestIdOrFail(requestId: number): Promise<CoughAudio> {
        return this.findOneOrFail({request_id: requestId});
    }
}