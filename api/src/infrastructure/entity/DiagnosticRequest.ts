import {Entity, PrimaryGeneratedColumn, Column, OneToOne, ManyToOne, Index, JoinColumn} from 'typeorm';
import {User} from './Users';
import {DiagnosticRequestStatus} from './DiagnostRequestStatus';
import {PatientInfo} from './PatientInfo';
import {CoughAudio} from './CoughAudio';
import {BreathAudio} from './BreathAudio';
import {SpeechAudio} from './SpeechAudio';
import {DiagnosticReport} from './DiagnosticReport';
import {CoughCharacteristics} from './CoughCharacteristic';

@Entity('diagnostic_requests')
export class DiagnosticRequest {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column()
    user_id: number

    @ManyToOne((type) => DiagnosticRequestStatus, (status) => status.id)
    @JoinColumn({name: 'status_id'})
    status: DiagnosticRequestStatus

    @Column()
    status_id: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    dateCreated: Date

    @Column({ 
        type: 'float',
        nullable: true
    })
    location_latitude: number

    @Column({
        type: 'float',
        nullable: true
    })
    location_longitude: number

    @Column({nullable: true})
    language: string

    @OneToOne((type) => PatientInfo, (patient) => patient.request)
    patient_info: PatientInfo

    @OneToOne((type) => CoughAudio, (cough_audio) => cough_audio.request)
    cough_audio: CoughAudio

    @OneToOne((type) => BreathAudio, (breath_audio) => breath_audio.request)
    breath_audio: BreathAudio

    @OneToOne((type) => SpeechAudio, (speech_audio) => speech_audio.request)
    speech_audio: SpeechAudio

    @OneToOne((type) => DiagnosticReport, (report) => report.request)
    diagnostic_report: DiagnosticReport

    @OneToOne((type) => CoughCharacteristics, (cough) => cough.request)
    cough_characteristics: CoughCharacteristics
}
