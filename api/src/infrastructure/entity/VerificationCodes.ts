import {Entity, PrimaryGeneratedColumn, Column, ManyToOne, Index, JoinColumn} from 'typeorm';
import {User} from './Users';

@Entity('verification_codes')
export class VerificationCodes {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column({nullable: false})
    user_id: number

    @Column({nullable: false})
    code: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    date_created: Date

    @Column('timestamp with time zone', {nullable: false})
    date_expired: Date

    @Column({default: false})
    is_verified: boolean;
}