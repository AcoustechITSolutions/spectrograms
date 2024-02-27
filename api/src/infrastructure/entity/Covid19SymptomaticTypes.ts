import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {Covid19SymptomaticTypes as DomainCovid19SymptomaticTypes} from '../../domain/Covid19Types';

@Entity('covid19_symptomatic_types')
export class Covid19SymptomaticTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: DomainCovid19SymptomaticTypes,
        unique: true,
    })
    symptomatic_type: DomainCovid19SymptomaticTypes
}
