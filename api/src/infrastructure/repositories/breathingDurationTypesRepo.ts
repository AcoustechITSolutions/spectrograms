import {EntityRepository, Repository} from 'typeorm';
import {BreathingDurationTypes} from '../entity/BreathingDurationTypes';

@EntityRepository(BreathingDurationTypes)
export class BreahtingDurationTypesRepository extends Repository<BreathingDurationTypes> {
    public async findByStringOrFail(type: string): Promise<BreathingDurationTypes> {
        return this.findOneOrFail({where: {duration_type: type}});
    }
}
