import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {BreathingDifficulty as BreathingDifficultyDomain} from '../../domain/BreathingCharacteristics';

@Entity('breathing_difficulty_types')
export class BreathingDifficultyTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: BreathingDifficultyDomain,
        unique: true,
    })
    difficulty_type: BreathingDifficultyDomain
}
