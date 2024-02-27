import {Entity, PrimaryGeneratedColumn, Column, JoinColumn, ManyToOne, OneToOne, Index} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';
import {GenderTypes} from './GenderTypes';
import {BotUsers} from './BotUsers';
import {TelegramDatasetRequestStatus} from './TelegramDatasetRequestStatus';

@Entity('tg_dataset_request')
export class TelegramDatasetRequest {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DatasetRequest, req => req.id, {nullable: true})
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column({nullable: true})
    request_id: number

    @Index()
    @ManyToOne((type) => BotUsers, user => user.chat_id,)
    @JoinColumn({name: 'chat_id', referencedColumnName: 'chat_id'})
    user: BotUsers

    @Column()
    chat_id: number

    @ManyToOne((type) => TelegramDatasetRequestStatus, (status) => status.id)
    @JoinColumn({name: 'status_id'})
    status: TelegramDatasetRequestStatus

    @Column()
    status_id: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    dateCreated: Date

    @Column({nullable: true})
    age: number

    @ManyToOne((type) => GenderTypes, (gender) => gender.id, {nullable: true})
    @JoinColumn({name: 'gender_id'})
    gender: GenderTypes

    @Column({nullable: true})
    gender_id: number

    @Column({nullable: true})
    is_smoking: boolean

    @Column({nullable: true})
    is_covid: boolean

    @Column({nullable: true})
    is_disease: boolean

    @Column({nullable: true})
    disease_name: string

    @Column({nullable: true})
    cough_audio_path: string

    @Column({nullable: true})
    is_forced: boolean
}
