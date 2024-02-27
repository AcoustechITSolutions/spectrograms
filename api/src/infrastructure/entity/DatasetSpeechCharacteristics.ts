import {Entity, PrimaryGeneratedColumn, Column, Index, OneToOne, JoinColumn} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';

@Entity('dataset_speech_characteristics')
export class DatasetSpeechCharacteristics {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DatasetRequest, (request) => request.id)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @Column({default: ''})
    commentary: string
}
