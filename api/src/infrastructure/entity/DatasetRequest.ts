import {Entity, PrimaryGeneratedColumn, Column, OneToOne, ManyToOne, Index, JoinColumn, OneToMany} from 'typeorm';
import {User} from './Users';
import {DatasetAudioInfo} from './DatasetAudioInfo';
import {DatasetBreathingCharacteristics} from './DatasetBreathingCharacteristics';
import {DatasetCoughCharacteristics} from './DatasetCoughCharacteristics';
import {DatasetPatientDetails} from './DatasetPatientDetails';
import {DatasetPatientDiseases} from './DatasetPatientDiseases';
import {DatasetRequestStatus} from './DatasetRequestStatus';
import {DatasetSpeechCharacteristics} from './DatasetSpeechCharacteristics';
import {DatasetMarkingStatus as EntityDatasetMarkingStatus} from './DatasetMarkingStatus';
import {DatasetBreathingGeneralInfo} from './DatasetBreathingGeneralInfo';

@Entity('dataset_request')
export class DatasetRequest {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column({
        default: true,
    })
    is_visible: boolean

    @Column()
    user_id: number

    @ManyToOne((type) => DatasetRequestStatus, (status) => status.id)
    @JoinColumn({name: 'status_id'})
    status: DatasetRequestStatus

    @Column()
    status_id: number

    @ManyToOne((type) => EntityDatasetMarkingStatus, (status) => status.marking_status)
    @JoinColumn({name: 'marking_status_id'})
    marking_status: EntityDatasetMarkingStatus

    @Column()
    marking_status_id: number

    @ManyToOne((type) => EntityDatasetMarkingStatus, (status) => status.marking_status)
    @JoinColumn({name: 'doctor_status_id'})
    doctor_status: EntityDatasetMarkingStatus

    @Column()
    doctor_status_id: number

    @Column('timestamp with time zone', {nullable: false, default: () => 'CURRENT_TIMESTAMP'})
    date_created: Date

    @Column()
    privacy_eula_version: number

    @OneToOne(type => DatasetBreathingGeneralInfo, breathingGeneralInfo => breathingGeneralInfo.request)
    breathing_general_info: DatasetBreathingGeneralInfo

    @OneToMany((type) => DatasetAudioInfo, (audioInfo) => audioInfo.request)
    audio_info: DatasetAudioInfo[]

    @OneToOne((type) => DatasetSpeechCharacteristics, (speechCharacteristics) => speechCharacteristics.request)
    speech_characteristics: DatasetSpeechCharacteristics

    @OneToMany((type) => DatasetBreathingCharacteristics, (breathingCharacteristics) => breathingCharacteristics.request)
    breathing_characteristics: DatasetBreathingCharacteristics[]

    @OneToOne((type) => DatasetCoughCharacteristics, (coughCharacteristics) => coughCharacteristics.request)
    cough_characteristics: DatasetCoughCharacteristics

    @OneToOne((type) => DatasetPatientDetails, (patientDetails) => patientDetails.request)
    patient_details: DatasetPatientDetails

    @OneToOne((type) => DatasetPatientDiseases, (patientDiseases) => patientDiseases.request)
    patient_diseases: DatasetPatientDiseases
}
