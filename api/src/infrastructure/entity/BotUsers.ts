import {Entity, PrimaryGeneratedColumn, Column, Index} from 'typeorm';

@Entity('bot_users')
export class BotUsers {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @Column({nullable: false, unique: true})
    chat_id: number

    @Column({default: 'ru'})
    report_language: string
}
