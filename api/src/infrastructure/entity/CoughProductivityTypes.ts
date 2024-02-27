import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {CoughProductivity} from '../../domain/CoughTypes';

@Entity('cough_productivity_types')
export class CoughProductivityTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: CoughProductivity,
        unique: true,
    })
    productivity_type: CoughProductivity
}
