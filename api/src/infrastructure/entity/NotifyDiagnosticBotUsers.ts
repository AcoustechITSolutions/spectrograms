import {Entity, PrimaryGeneratedColumn, Column, Index} from 'typeorm';

@Entity('notify_diagnostic_bot_users')
export class NotifyDiagnosticBotUsers {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @Column({nullable: false, unique: true})
    chat_id: number

    @Column({default: 'ru'})
    language: string
}
