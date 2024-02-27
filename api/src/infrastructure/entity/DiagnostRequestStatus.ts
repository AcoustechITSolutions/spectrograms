import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus} from '../../domain/RequestStatus';

@Entity('diagnostic_request_status')
export class DiagnosticRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainDiagnosticRequestStatus,
        unique: true,
    })
    request_status: DomainDiagnosticRequestStatus
}
