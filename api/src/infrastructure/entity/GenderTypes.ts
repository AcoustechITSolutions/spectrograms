import {Column,   Entity, PrimaryGeneratedColumn} from 'typeorm';
import {Gender as GenderTypesDomain} from '../../domain/Gender';

@Entity('gender_types')
export class GenderTypes {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: GenderTypesDomain,
        unique: true,
    })
    gender_type: GenderTypesDomain
}
