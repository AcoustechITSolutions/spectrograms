import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {DiagnosisTypes as domainTypes} from '../../domain/DiagnosisTypes';

@Entity('diagnosis_types')
export class DiagnosisTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: domainTypes,
        unique: true,
    })
    diagnosis_type: domainTypes
}
