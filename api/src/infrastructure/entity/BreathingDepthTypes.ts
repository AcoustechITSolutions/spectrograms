import {PrimaryGeneratedColumn, Column, Entity} from 'typeorm';
import {BreathingDepth as BreathingDepthDomain} from '../../domain/BreathingCharacteristics';

@Entity('breathing_depth_types')
export class BreathingDepthTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: BreathingDepthDomain,
        unique: true,
    })
    depth_type: BreathingDepthDomain
}
