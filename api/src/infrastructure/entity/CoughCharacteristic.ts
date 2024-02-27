import {PrimaryGeneratedColumn, Column, OneToOne, JoinColumn, Entity, ManyToOne} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';
import {CoughIntensityTypes} from './CoughIntensityTypes';
import {CoughProductivityTypes} from './CoughProductivityTypes';

@Entity('cough_characteristics')
export class CoughCharacteristics {
    @PrimaryGeneratedColumn()
    id: number

    @OneToOne((type) => DiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: DiagnosticRequest

    @Column()
    request_id: number

    @Column({nullable: true})
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
}
