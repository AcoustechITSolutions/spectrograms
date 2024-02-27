import {PrimaryGeneratedColumn, Entity, Column} from 'typeorm';
import {BreathingTypes as BreathingTypesDomain} from '../../domain/BreathingCharacteristics';

@Entity('breathing_types')
export class BreathingTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: BreathingTypesDomain,
        unique: true,
    })
    breathing_type: BreathingTypesDomain
}
