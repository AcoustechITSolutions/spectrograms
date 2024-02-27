import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DatasetEpisodesTypes as DomainDatasetEpisodesTypes} from '../../domain/DatasetEpisodesTypes';

@Entity('dataset_episodes_types')
export class DatasetEpisodesTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainDatasetEpisodesTypes,
        unique: true
    })
    episode_type: DomainDatasetEpisodesTypes
}