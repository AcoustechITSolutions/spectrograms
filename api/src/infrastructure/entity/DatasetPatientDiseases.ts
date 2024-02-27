import {Column, OneToOne, JoinColumn, Entity, PrimaryGeneratedColumn, ManyToOne, Index} from 'typeorm';
import {DiseaseTypes} from './DiseaseTypes';
import {AcuteCoughTypes} from './AcuteCoughTypes';
import {ChronicCoughTypes} from './ChronicCoughTypes';
import {DatasetRequest} from './DatasetRequest';
import {Covid19SymptomaticTypes} from './Covid19SymptomaticTypes';

@Entity('dataset_patient_diseases')
export class DatasetPatientDiseases {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DatasetRequest, (request) => request.id)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @Column()
    sick_days: number

    @ManyToOne((type) => Covid19SymptomaticTypes, (type) => type.id, {
        nullable: true,
    })
    @JoinColumn({
        name: 'covid19_symptomatic_type_id',
    })
    covid19_symptomatic_type: Covid19SymptomaticTypes

    @Column({
        nullable: true,
    })
    covid19_symptomatic_type_id: number

    @Column({
        nullable: true,
    })
    other_disease_name: string

    @ManyToOne((type) => DiseaseTypes, {
        nullable: true,
    })
    @JoinColumn({
        name: 'disease_type_id',
    })
    disease_type: DiseaseTypes

    @Column({
        nullable: true,
    })
    disease_type_id: number

    @ManyToOne((type) => AcuteCoughTypes, {
        nullable: true,
    })
    @JoinColumn({
        name: 'acute_cough_types_id',
    })
    acute_cough_types: AcuteCoughTypes

    @Column({
        nullable: true,
    })
    acute_cough_types_id: number

    @ManyToOne((type) => ChronicCoughTypes, {
        nullable: true,
    })
    @JoinColumn({
        name: 'chronic_cough_types_id',
    })
    chronic_cough_types: ChronicCoughTypes

    @Column({
        nullable: true,
    })
    chronic_cough_types_id: number
}
