import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {HWRequestStatus as HWRequestStatusDomain} from '../../domain/RequestStatus';

@Entity('hw_diagnostic_request_status')
export class HWRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: HWRequestStatusDomain,
        unique: true,
    })
    request_status: HWRequestStatusDomain
}
