import {Column, OneToOne, JoinColumn, Entity, PrimaryGeneratedColumn, Index} from 'typeorm';
import {DiagnosticRequest} from './DiagnosticRequest';
import {Gender} from '../../domain/Gender';

@Entity('patient_info')
export class PatientInfo {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => DiagnosticRequest)
    @JoinColumn({name: 'request_id'})
    request: DiagnosticRequest

    @Column()
    request_id: number

    @Column({nullable: true})
    age: number

    @Column({
        type: 'enum',
        enum: ['male', 'female'],
        nullable: true
    })
    gender: Gender

    @Column({nullable: true})
    is_smoking: boolean

    @Column({nullable: true})
    sick_days: number

    @Column({nullable: true})
    identifier: string

    @Column({
        length: 255,
        nullable: true,
    })
    photo_path: string
}
