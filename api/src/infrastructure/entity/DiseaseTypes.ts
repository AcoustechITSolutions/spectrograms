import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {DiseaseTypes as domainTypes} from '../../domain/DiseaseTypes';

@Entity('disease_types')
export class DiseaseTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: domainTypes,
        unique: true,
    })
    disease_type: domainTypes
}
