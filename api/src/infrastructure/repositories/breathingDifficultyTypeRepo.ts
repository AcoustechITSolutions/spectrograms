import {EntityRepository, Repository} from 'typeorm';
import {BreathingDifficultyTypes} from '../entity/BreathingDifficultyTypes';

@EntityRepository(BreathingDifficultyTypes)
export class BreathingDifficultyTypesRepository extends Repository<BreathingDifficultyTypes> {
    public async findByStringOrFail(type: string): Promise<BreathingDifficultyTypes> {
        return this.findOneOrFail({where: {difficulty_type: type}});
    }
}
