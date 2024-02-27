import {PrimaryGeneratedColumn, Column, OneToOne, JoinColumn, Entity, ManyToOne, Index} from 'typeorm';
import {DatasetRequest} from './DatasetRequest';
import {CoughIntensityTypes} from './CoughIntensityTypes';
import {CoughProductivityTypes} from './CoughProductivityTypes';

@Entity('dataset_cough_characteristics')
export class DatasetCoughCharacteristics {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DatasetRequest)
    @JoinColumn({name: 'request_id'})
    request: DatasetRequest

    @Column()
    request_id: number

    @Column()
    is_forced: boolean

    @ManyToOne((type) => CoughProductivityTypes, {nullable: true})
    @JoinColumn({name: 'productivity_id'})
    productivity: CoughProductivityTypes

    @Column({nullable: true})
    productivity_id: number

    @ManyToOne((type) => CoughIntensityTypes, {nullable: true})
    @JoinColumn({name: 'intensity_id'})
    intensity: CoughIntensityTypes

    @Column({nullable: true})
    intensity_id: number

    @Column({
        nullable: true,
    })
    symptom_duration: number

    @Column({
        nullable: true,
    })
    commentary: string
}
