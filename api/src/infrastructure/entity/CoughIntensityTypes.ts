import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {CoughIntensity} from '../../domain/CoughTypes';

@Entity('cough_intensity_types')
export class CoughIntensityTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: CoughIntensity,
        unique: true,
    })
    intensity_type: CoughIntensity
}
