import {Entity, PrimaryGeneratedColumn, Column,  ManyToOne, JoinColumn} from 'typeorm';
import {DatasetAudioInfo} from './DatasetAudioInfo';
import {DatasetEpisodesTypes} from './DatasetEpisodesTypes';

@Entity('dataset_audio_espisodes')
export class DatasetAudioEpisodes {
    @PrimaryGeneratedColumn()
    id: number

    @ManyToOne((type) => DatasetAudioInfo, (info) => info.id)
    @JoinColumn({name: 'audio_info_id'})
    audio_info: DatasetAudioInfo

    @ManyToOne(type => DatasetEpisodesTypes, type => type.id)
    @JoinColumn({name: 'episode_type_id'})
    episode_type: DatasetEpisodesTypes

    @Column({default: 3})
    episode_type_id: number

    @Column()
    audio_info_id: number

    @Column({
        type: 'float',
    })
    start: number

    @Column({
        type: 'float',
    })
    end: number
}
