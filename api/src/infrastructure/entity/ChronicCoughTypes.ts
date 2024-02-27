import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {ChronicCoughTypes as domainTypes} from '../../domain/DiseaseTypes';

@Entity('chronic_cough_types')
export class ChronicCoughTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: domainTypes,
        unique: true,
    })
    chronic_cough_type: domainTypes
}
