import {Entity, PrimaryGeneratedColumn, Column,  ManyToOne, Index, JoinColumn} from 'typeorm';
import {HWRequestStatus} from './HWRequestStatus';
import {User} from './Users';

@Entity('hw_diagnostic_request')
export class HWDiagnosticRequest {
    @PrimaryGeneratedColumn()
    id: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    date_created: Date

    @Index()
    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column()
    user_id: number

    @Index()
    @ManyToOne((type) => HWRequestStatus, (status) => status.id)
    @JoinColumn({name: 'status_id'})
    status: HWRequestStatus

    @Column()
    status_id: number
}
