import {Entity, PrimaryGeneratedColumn, Column, OneToOne, JoinColumn} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';

@Entity('breath_audio')
export class BreathAudio {
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
