import {Entity, PrimaryGeneratedColumn, Column, OneToOne, JoinColumn} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';

@Entity('cough_audio')
export class CoughAudio {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        length: 255,
    })
    file_path: string

    @OneToOne((type) => DiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: DiagnosticRequest

    @Column()
    request_id: number

    @Column({
        type: 'int',
        nullable: true,
    })
    samplerate: number

    @Column({
        type: 'float',
        nullable: true,
    })
    duration: number

    @Column({
        length: 255,
        nullable: true
    })
    spectrogram_path: string
}
