import {Entity, PrimaryGeneratedColumn, Column, JoinColumn, ManyToOne, OneToOne, Index} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';
import {GenderTypes} from './GenderTypes';
import {BotUsers} from './BotUsers';
import {TelegramDiagnosticRequestStatus} from './TelegramDiagnosticRequestStatus';

@Entity('tg_diagnostic_request')
export class TelegramDiagnosticRequest {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DiagnosticRequest, req => req.id, {nullable: true})
    @JoinColumn({name: 'request_id'})
    request: DiagnosticRequest

    @Column({nullable: true})
    request_id: number

    @Index()
    @ManyToOne((type) => BotUsers, user => user.chat_id)
    @JoinColumn({name: 'chat_id', referencedColumnName: 'chat_id'})
    user: BotUsers

    @Column()
    chat_id: number

    @ManyToOne((type) => TelegramDiagnosticRequestStatus, (status) => status.id)
    @JoinColumn({name: 'status_id'})
    status: TelegramDiagnosticRequestStatus

    @Column()
    status_id: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    dateCreated: Date

    @Column('timestamp with time zone', {nullable: true})
    date_finished: Date

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
    cough_audio_path: string

    @Column({nullable: true})
    is_forced: boolean
}
