import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DatasetAudioTypes as DatasetAudioTypesDomain} from '../../domain/DatasetAudio';

@Entity('dataset_audio_types')
export class DatasetAudioTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DatasetAudioTypesDomain,
        unique: true,
    })
    audio_type: DatasetAudioTypesDomain
}
