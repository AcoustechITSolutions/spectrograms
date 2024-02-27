import {Column, Entity, PrimaryGeneratedColumn, ManyToMany} from 'typeorm';
import {UserRoleTypes} from '../../domain/UserRoles';
import {User} from './Users';

@Entity('roles')
export class Roles {
    @PrimaryGeneratedColumn()
    id: number

    @Column({
        type: 'enum',
        enum: UserRoleTypes,
        unique: true,
    })
    role: UserRoleTypes

    @ManyToMany(type => User, user => user.roles)
    users: User[];
}
