import {Entity, PrimaryGeneratedColumn, Column, OneToOne, JoinColumn} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';

@Entity('speech_audio')
export class SpeechAudio {
    @PrimaryGeneratedColumn()
    id: number

    @Column()
    file_path: string

    @OneToOne((type) => DiagnosticRequest)
    @JoinColumn({
        name: 'request_id',
    })
    request: DiagnosticRequest

    @Column()
    request_id: number
}
