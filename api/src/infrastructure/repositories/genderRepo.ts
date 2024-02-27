import {EntityRepository, Repository} from 'typeorm';
import {GenderTypes} from '../entity/GenderTypes';

@EntityRepository(GenderTypes)
export class GenderTypesRepository extends Repository<GenderTypes> {
    public async findByStringOrFail(type: string): Promise<GenderTypes> {
        return this.findOneOrFail({where: {gender_type: type}});
    }
}