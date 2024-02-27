import {Entity, PrimaryGeneratedColumn, Column} from 'typeorm';
import {NoiseTypes as NoiseTypesDomain} from '../../domain/NoiseTypes';

@Entity('noise_types')
export class NoiseTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: NoiseTypesDomain,
        unique: true,
    })
    noise_type: NoiseTypesDomain
}