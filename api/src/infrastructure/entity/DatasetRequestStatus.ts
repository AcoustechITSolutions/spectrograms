import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DatasetRequestStatus as DomainDatasetRequestStatus} from '../../domain/RequestStatus';

@Entity('dataset_request_status')
export class DatasetRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainDatasetRequestStatus,
        unique: true,
    })
    request_status: DomainDatasetRequestStatus
}
