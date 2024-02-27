import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {TelegramDatasetRequestStatus as DomainTgDataStatus} from '../../domain/RequestStatus';

@Entity('tg_dataset_requests_status')
export class TelegramDatasetRequestStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainTgDataStatus,
        unique: true
    })
    request_status: DomainTgDataStatus
}
