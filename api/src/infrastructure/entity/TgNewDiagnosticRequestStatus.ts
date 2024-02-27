import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {TgNewDiagnosticRequestStatus as DomainTgStatus} from '../../domain/RequestStatus';

@Entity('tg_new_diagnostic_request_status')
export class TgNewDiagnosticRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainTgStatus,
        unique: true
    })
    request_status: DomainTgStatus
}
