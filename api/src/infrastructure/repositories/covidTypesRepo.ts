import {EntityRepository, Repository} from 'typeorm';
import {Covid19SymptomaticTypes} from '../entity/Covid19SymptomaticTypes';

@EntityRepository(Covid19SymptomaticTypes)
export class CovidTypesRepository extends Repository<Covid19SymptomaticTypes> {
    public async findByStringOrFail(type: string): Promise<Covid19SymptomaticTypes> {
        return this.findOneOrFail({where: {symptomatic_type: type}});
    }
}