import {PrimaryGeneratedColumn, Column, OneToOne, Entity, JoinColumn, Index} from 'typeorm';

import {HWDiagnosticRequest} from './HWDiagnosticRequest';

@Entity('hw_diagnostic_report')
export class HWDiagnosticReport {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => HWDiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: HWDiagnosticRequest

    @Column()
    request_id: number

    @Column({
        type: 'float',
        nullable: true,
    })
    diagnosis_probability: number
}
