import {PrimaryGeneratedColumn, Entity, Column} from 'typeorm';
import {BreathingDuration as BreathingDurationDomain} from '../../domain/BreathingCharacteristics';

@Entity('breathing_duration_types')
export class BreathingDurationTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: BreathingDurationDomain,
        unique: true,
    })
    duration_type: BreathingDurationDomain
}
