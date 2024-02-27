import {Column, OneToOne, JoinColumn, Entity, PrimaryGeneratedColumn,    ManyToOne, Index} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';
import {GenderTypes} from './GenderTypes';

@Entity('dataset_patient_details')
export class DatasetPatientDetails {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DatasetRequest)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @Column()
    age: number

    @ManyToOne((type) => GenderTypes, (gender) => gender.id)
    @JoinColumn({name: 'gender_type_id'})
    gender: GenderTypes

    @Column()
    gender_type_id: number

    @Column()
    identifier: string

    @Column()
    is_smoking: boolean
}
