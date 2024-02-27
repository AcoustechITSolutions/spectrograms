import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DatasetMarkingStatus as MarkingStatusDomain} from '../../domain/DatasetMarkingStatus';

@Entity('dataset_marking_status')
export class DatasetMarkingStatus {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: MarkingStatusDomain,
        unique: true,
    })
    marking_status: MarkingStatusDomain
}
