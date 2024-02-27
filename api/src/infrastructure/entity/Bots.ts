import {Column, Entity, PrimaryGeneratedColumn} from 'typeorm';
import {Bots as BotsDomain} from '../../domain/Bots';

@Entity('bots')
export class Bots {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: BotsDomain,
        unique: true,
    })
    bot_name: BotsDomain
}
