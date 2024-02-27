import {Entity, PrimaryGeneratedColumn, Column, OneToOne, JoinColumn, Index} from 'typeorm';
import {HWDiagnosticRequest} from './HWDiagnosticRequest';

@Entity('hw_cough_audio')
export class HWCoughAudio {
    @PrimaryGeneratedColumn()
    id: number

    @Column()
    file_path: string

    @Index()
    @OneToOne((type) => HWDiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: HWDiagnosticRequest

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
}
