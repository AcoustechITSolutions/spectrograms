import {Column, Entity, PrimaryGeneratedColumn} from 'typeorm';
import {Payments as PaymentTypesDomain} from '../../domain/Payments';

@Entity('payment_types')
export class PaymentTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: PaymentTypesDomain,
        unique: true,
    })
    payment_type: PaymentTypesDomain
}
