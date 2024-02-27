import {Column, OneToOne, JoinColumn, Entity, PrimaryGeneratedColumn, Index, ManyToOne} from 'typeorm';
import {User} from './Users';
import {GenderTypes} from './GenderTypes';

@Entity('personal_data')
export class PersonalData {
    @PrimaryGeneratedColumn()
    id: number

    @Index()
    @OneToOne((type) => User)
    @JoinColumn({name: 'user_id'})
    user: User

    @Column({nullable: false})
    user_id: number

    @Column({nullable: true})
    identifier: string

    @Column({nullable: true})
    age: number

    @ManyToOne((type) => GenderTypes, (gender) => gender.id, {
        nullable: true, 
        eager: true
    })
    @JoinColumn({name: 'gender_id'})
    gender: GenderTypes

    @Column({nullable: true})
    gender_id: number

    @Column({nullable: true})
    is_smoking: boolean

    @Column({nullable: true, type: 'varchar'})
    voice_embedding: string

    @Column({nullable: true, type: 'varchar'})
    voice_audio_path: string
}
