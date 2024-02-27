import {
    Entity,
    PrimaryGeneratedColumn,
    Column,
    CreateDateColumn,
    ManyToOne,
    Index,
    JoinColumn, 
} from 'typeorm';
import {User} from './Users';

@Entity('refresh_token')
export class RefreshToken {
    @PrimaryGeneratedColumn('uuid')
    id: string;

    @Index()
    @ManyToOne((type) => User, (user) => user.refresh_tokens)
    @JoinColumn({name: 'user_id'})
    user: User;

    @Column({nullable: false, unique: true, length: 255, type: 'varchar'})
    jwt_id: string;

    @Column()
    expires_date: Date;

    @CreateDateColumn()
    created_date: Date;
}
