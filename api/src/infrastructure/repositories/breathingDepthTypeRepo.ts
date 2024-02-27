import {EntityRepository, Repository} from 'typeorm';
import {BreathingDepthTypes} from '../entity/BreathingDepthTypes';

@EntityRepository(BreathingDepthTypes)
export class BreathingDepthTypesRepository extends Repository<BreathingDepthTypes> {
    public async findByStringOrFail(type: string): Promise<BreathingDepthTypes> {
        return this.findOneOrFail({where: {depth_type: type}});
    }
}
