import {PrimaryGeneratedColumn, Column,  JoinColumn, Entity, ManyToOne, Index} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';
import {BreathingTypes} from './BreathingTypes';
import {BreathingDepthTypes} from './BreathingDepthTypes';
import {BreathingDifficultyTypes} from './BreathingDifficultyTypes';
import {BreathingDurationTypes} from './BreathingDurationTypes';

@Entity('dataset_breathing_characteristics')
export class DatasetBreathingCharacteristics {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @ManyToOne((type) => DatasetRequest, (request) => request.id)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @ManyToOne((type) => BreathingTypes, (breathingType) => breathingType.id)
    @JoinColumn({name: 'breathing_type_id'})
    breathing_type: BreathingTypes

    @Column()
    breathing_type_id: number

    @ManyToOne((type) => BreathingDepthTypes, (depthType) => depthType.id, {nullable: true})
    @JoinColumn({name: 'depth_type_id'})
    depth_type: BreathingDepthTypes

    @Column({nullable: true})
    depth_type_id: number

    @ManyToOne((type) => BreathingDifficultyTypes, (difficultyType) => difficultyType.id, {nullable: true})
    @JoinColumn({name: 'difficulty_type_id'})
    difficulty_type: BreathingDifficultyTypes

    @Column({nullable: true})
    difficulty_type_id: number

    @ManyToOne((type) => BreathingDurationTypes, (durationType) => durationType.id, {nullable: true})
    @JoinColumn({name: 'duration_type_id'})
    duration_type: BreathingDurationTypes

    @Column({nullable: true})
    duration_type_id: number
}
