import {Column, OneToOne, JoinColumn, Entity, PrimaryGeneratedColumn, Index} from 'typeorm';
import {User} from './Users';

@Entity('user_subscriptions')
export class UserSubscriptions {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => User)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column()
    user_id: number

    @Column()
    diagnostics_left: number
}
