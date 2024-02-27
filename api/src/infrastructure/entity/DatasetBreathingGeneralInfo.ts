import {Column, Entity, Index, JoinColumn, PrimaryGeneratedColumn, OneToOne} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';

@Entity('dataset_breathing_general_info')
export class DatasetBreathingGeneralInfo {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne(type => DatasetRequest, request => request.id)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @Column({nullable: true})
    commentary: string
}
