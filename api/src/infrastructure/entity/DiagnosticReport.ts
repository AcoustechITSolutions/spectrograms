import {PrimaryGeneratedColumn, Column, OneToOne, Entity, JoinColumn, Index, ManyToOne, Generated} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';
import {User} from './Users';
import {DiagnosticReportEpisodes} from './DiagnosisReportEpisodes';
import {DiagnosisTypes} from './DiagnosisTypes';

@Entity('diagnostic_report')
export class DiagnosticReport {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: DiagnosticRequest

    @Column()
    request_id: number

    @Index()
    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'user_id'})
    user: User

    @OneToOne((type) => DiagnosticReportEpisodes)
    episodes: DiagnosticReportEpisodes

    @Column()
    user_id: number

    @Column({
        nullable: true,
    })
    commentary: string

    @Column({
        default: false,
    })
    is_confirmed: boolean

    @Column({
        default: true,
    })
    is_visible: boolean

    @ManyToOne((type) => DiagnosisTypes, {nullable: true})
    @JoinColumn({name: 'diagnosis_id'})
    diagnosis: DiagnosisTypes

    @Column({nullable: true})
    diagnosis_id: number

    @Column({
        type: 'float',
        nullable: true,
    })
    diagnosis_probability: number

    @Column()
    @Generated('uuid')
    qr_code_token: string

    @Column({
        nullable: true,
    })
    nationality: string

    @Column({
        nullable: true,
    })
    is_pcr_positive: boolean
}
