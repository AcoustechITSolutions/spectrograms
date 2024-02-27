import {
    Entity,
    PrimaryColumn,
    ManyToOne,
    Index,
    JoinColumn, 
} from 'typeorm';
import {User} from './Users';

@Entity('doctors_patients')
@Index(['doctor_id', 'patient_id'], { unique: true })
export class DoctorsPatients {
    @PrimaryColumn({nullable: false})
    doctor_id: number;

    @PrimaryColumn({nullable: false})
    patient_id: number;

    @ManyToOne((type) => User, (user) => user.patients)
    @JoinColumn({name: 'doctor_id'})
    doctor: User;

    @ManyToOne((type) => User, (user) => user.id)
    @JoinColumn({name: 'patient_id'})
    patient: User;
}