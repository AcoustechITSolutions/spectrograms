import {Entity, PrimaryGeneratedColumn, Column, ManyToOne, JoinColumn, Index, OneToMany} from 'typeorm';
import {DatasetAudioTypes} from './DatasetAudioTypes';
import {NoiseTypes} from './NoiseTypes';
import {DatasetRequest} from './DatasetRequest';

import {DatasetAudioEpisodes} from './DatasetAudioEpisodes';

@Entity('dataset_audio_info')
export class DatasetAudioInfo {
    @PrimaryGeneratedColumn()
    id: number

    @Column({nullable: true})
    is_representative: boolean

    @Column({nullable: true})
    is_representative_scientist: boolean

    @Column({
        default: false,
    })
    is_validation_audio: boolean

    @Index()
    @ManyToOne((type) => DatasetRequest, (request) => request.id)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @ManyToOne((type) => DatasetAudioTypes, (type) => type.id)
    @JoinColumn({name: 'audio_type_id'})
    audio_type: DatasetAudioTypes

    @Column()
    audio_type_id: number

    @Column({
        nullable: true,
    })
    samplerate: number

    @Column()
    audio_path: string

    @Column({
        nullable: true,
    })
    spectrogram_path: string

    @Column({
        type: 'float',
        nullable: true,
    })
    audio_duration: number

    @Column({
        default: false
    })
    is_marked: boolean

    @Column({
        default: false
    })
    is_marked_scientist: boolean

    @OneToMany((type) => DatasetAudioEpisodes, (episodes) => episodes.audio_info)
    episodes_duration: DatasetAudioEpisodes[]

    @ManyToOne((type) => NoiseTypes, {nullable: true})
    @JoinColumn({name: 'noise_type_id'})
    noise_type: NoiseTypes

    @Column({nullable: true})
    noise_type_id: number
}
