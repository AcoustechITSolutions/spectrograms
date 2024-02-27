import {EntityRepository, Repository} from 'typeorm';
import {PaymentTypes} from '../entity/PaymentTypes';

@EntityRepository(PaymentTypes)
export class PaymentTypesRepository extends Repository<PaymentTypes> {
    public async findByStringOrFail(type: string): Promise<PaymentTypes> {
        return this.findOneOrFail({where: {payment_type: type}});
    }
}