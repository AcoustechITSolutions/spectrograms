import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {TelegramDiagnosticRequestStatus as DomainTgStatus} from '../../domain/RequestStatus';

@Entity('tg_diagnostic_requests_status')
export class TelegramDiagnosticRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainTgStatus,
        unique: true
    })
    request_status: DomainTgStatus
}
