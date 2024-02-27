import {Entity, PrimaryGeneratedColumn, Column, ManyToMany, JoinTable, OneToMany} from 'typeorm';
import {Roles} from './Roles';
import {DiagnosticRequest} from './DiagnosticRequest';
import {RefreshToken} from './RefreshToken';
import {DoctorsPatients} from './DoctorsPatients';

@Entity('users')
export class User {
    @PrimaryGeneratedColumn()
    id: number;

    @Column({nullable: false, unique: true, length: 255, type: 'varchar'})
    login: string;

    @Column({nullable: true, unique: true, length: 255, type: 'varchar'})
    email: string;

    @Column({nullable: true, length: 255, type: 'varchar'})
    password: string;

    @Column({nullable: true, unique: true})
    phone_number: string;

    @Column({nullable: true})
    comment: string;

    @Column({default: true})
    is_active: boolean;

    @Column({default: false})
    is_email_error: boolean;

    @Column({nullable: true})
    email_error_type: string;

    @Column({default: true})
    is_all_patients: boolean;

    @Column({default: false})
    is_check_healthy: boolean;

    @Column({default: true})
    is_check_covid: boolean;

    @Column({nullable: true})
    check_start: string;

    @Column({nullable: true})
    check_end: string;

    @Column({default: true})
    is_validate_cough: boolean;

    @ManyToMany((type) => Roles, role => role.users, {
        cascade: true,
        eager: true
    })
    @JoinTable()
    roles: Roles[];

    @OneToMany((type) => DiagnosticRequest, (request) => request.user)
    requests: DiagnosticRequest[];

    @OneToMany((type) => RefreshToken, (refreshToken) => refreshToken.user)
    refresh_tokens: RefreshToken;

    @OneToMany((type) => DoctorsPatients, (doctorsPatients) => doctorsPatients.doctor, {
        cascade: true
    })
    patients: DoctorsPatients[];
}
