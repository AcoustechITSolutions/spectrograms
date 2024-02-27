import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {AcuteCoughTypes as domainTypes} from '../../domain/DiseaseTypes';

@Entity('acute_cough_types')
export class AcuteCoughTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: domainTypes,
        unique: true,
    })
    acute_cough_types: domainTypes
}
