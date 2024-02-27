import {UserRoleTypes} from '../../domain/UserRoles';
import {EntityRepository, Repository} from 'typeorm';
import {User} from '../entity/Users';
import {DoctorsPatients} from '../entity/DoctorsPatients';

export const BOT_USER = 'telegram_bot';
export const DATASET_BOT_USER = 'dataset_bot';
export const DIAGNOSTIC_BOT_USER = 'diagnostic_bot';
export const MUUS_BOT_USER = 'muus_bot';

@EntityRepository(User)
export class UserRepository extends Repository<User> {
    public async findByLogin(login: string): Promise<User> {
        return this.findOne({
            where: {login: login}
        });
    }

    public async findByEmailOrNumber(login: string): Promise<User> {
        return this.findOne({
            where: [
                {email: login},
                {phone_number: login}
            ]
        });
    }

    public async findActiveByIdOrFail(user_id: number): Promise<User> {
        return this.findOneOrFail({
            relations: ['roles'],
            where: {
                id: user_id, is_active: true
            }
        })
    }
    
    public async findActiveByLoginOrEmailOrFail(
        login?: string,
        email?: string,
    ): Promise<User> {
        return this.findOneOrFail({
            // Needs roles for admin login check
            relations: ['roles'],
            where: [
                {login: login?.toLowerCase(), is_active: true},
                {email: email?.toLowerCase(), is_active: true}
            ]
        });
    }

    public async findPatientsByUserId(userId: number): Promise<string[]> {
        const patients = await this.createQueryBuilder('patient')
            .select('patient.login', 'login')
            .leftJoin(DoctorsPatients, 'patient_list', 'patient_list.patient_id = patient.id')
            .leftJoin(User, 'doctors', 'doctors.id = patient_list.doctor_id')
            .where('doctors.id = :user_id', {user_id: userId})
            .execute() as {login: string}[];   

        return patients.map((patient) => patient.login);
    }

    public async findDoctorsPhoneNumbers(): Promise<string[]> {
        const users = await this.createQueryBuilder('user')
            .select('user.phone_number')
            .leftJoinAndSelect('user.roles', 'roles')
            .where('roles.role = :doctor_role', {doctor_role: UserRoleTypes.EDIFIER})
            .andWhere('user.phone_number is not null')
            .getMany();
        return users.map((user) => user.phone_number);
    }

    public async findMaxId(): Promise<number | undefined> {
        const maxId = await this.createQueryBuilder('user')
            .select('max(user.id)', 'id')
            .getRawOne();
        return maxId.id;
    }

    public async findTelegramBotUserId(): Promise<number | undefined> {
        const user = await this.findByLogin(BOT_USER);
        return user?.id;
    }

    public async findDatasetBotUserId(): Promise<number | undefined> {
        const user = await this.findByLogin(DATASET_BOT_USER);
        return user?.id;
    }

    public async findDiagnosticBotUserId(): Promise<number | undefined> {
        const user = await this.findByLogin(DIAGNOSTIC_BOT_USER);
        return user?.id;
    }

    public async findMuusBotUserId(): Promise<number | undefined> {
        const user = await this.findByLogin(MUUS_BOT_USER);
        return user?.id;
    }
}
