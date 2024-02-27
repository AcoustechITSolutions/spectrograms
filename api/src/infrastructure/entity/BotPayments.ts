import {Entity, PrimaryGeneratedColumn, Column, JoinColumn, ManyToOne,  Index} from 'typeorm';
import {BotUsers} from './BotUsers';
import {Bots} from './Bots';
import {PaymentTypes} from './PaymentTypes';

@Entity('bot_payments')
export class BotPayments {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @ManyToOne((type) => BotUsers, (user) => user.chat_id, {nullable: false})
    @JoinColumn({name: 'chat_id', referencedColumnName: 'chat_id'})
    user: BotUsers

    @Column({nullable: false})
    chat_id: number

    @Index()
    @ManyToOne((type) => Bots, (bot) => bot.id, {nullable: false})
    @JoinColumn({name: 'bot_id'})
    bot: Bots

    @Column({nullable: false})
    bot_id: number

    @ManyToOne((type) => PaymentTypes, (payment) => payment.id, {nullable: false})
    @JoinColumn({name: 'payment_type_id'})
    payment_type: PaymentTypes

    @Column({nullable: false})
    payment_type_id: number

    @Column({default: false})
    is_active: boolean
}
