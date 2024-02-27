import {Entity, PrimaryGeneratedColumn, Column, JoinColumn, ManyToOne, Index} from 'typeorm';
import {TgNewDiagnosticRequest} from './TgNewDiagnosticRequest';
import {Bots} from './Bots';

@Entity('payonline_transactions')
export class PayonlineTransactions {
    @PrimaryGeneratedColumn()
    id: number

    @Column({nullable: true})
    transaction_id: number

    @Column('timestamp', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    date_created: Date

    @Column('timestamp', {nullable: true})
    date_updated: Date

    @Index()
    @ManyToOne((type) => TgNewDiagnosticRequest, req => req.id)
    @JoinColumn({name: 'request_id'})
    request: TgNewDiagnosticRequest

    @Column({nullable: false})
    request_id: number

    @Index()
    @ManyToOne((type) => Bots, bot => bot.id)
    @JoinColumn({name: 'bot_id'})
    bot: Bots

    @Column({nullable: false})
    bot_id: number

    @Column({nullable: false})
    amount: number

    @Column({nullable: false})
    currency: string

    @Column({default: false})
    is_confirmed: boolean
}
