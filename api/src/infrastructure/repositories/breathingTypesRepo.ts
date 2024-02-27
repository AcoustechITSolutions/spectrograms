import {EntityRepository, Repository} from 'typeorm';
import {BreathingTypes} from '../entity/BreathingTypes';

@EntityRepository(BreathingTypes)
export class BreathingTypesRepository extends Repository<BreathingTypes> {
    public async findByStringOrFail(type: string): Promise<BreathingTypes> {
        return this.findOneOrFail({where: {breathing_type: type}});
    }
}
