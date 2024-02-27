import {PrimaryGeneratedColumn, Column, OneToOne, Entity, JoinColumn, Index} from 'typeorm';
import {DiagnosticReport} from './DiagnosticReport';

@Entity('diagnosis_report_episodes')
export class DiagnosticReportEpisodes {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DiagnosticReport)
    @JoinColumn({name: 'report_id'})
    report: DiagnosticReport

    @Column()
    report_id: number

    @Column()
    episodes_count: number

    @Column({
        type: 'float',
        array: true,
    })
    duration_each: number[]

    @Column({
        type: 'float',
    })
    mean_duration: number

    @Column({
        type: 'float',
    })
    max_duration: number

    @Column({
        type: 'float',
    })
    min_duration: number

    @Column({
        type: 'float',
    })
    overall_duration: number
}
