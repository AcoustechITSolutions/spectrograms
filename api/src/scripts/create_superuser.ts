import {createConnection} from 'typeorm';
import {UserRoleTypes} from '../domain/UserRoles';
import {User} from '../infrastructure/entity/Users';
import {Roles} from '../infrastructure/entity/Roles';
import argon2 from 'argon2';

const createSuperUser = async () => {
    const connection = await createConnection({
        type: 'postgres',
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        password: process.env.DB_PASSWORD,
        username: process.env.DB_USER,
        database: process.env.DB_NAME,
        logging: true,
        entities: ['./dist/infrastructure/entity/*.js'],
        name: 'createSuperUser'
    });

    const adminRole = await connection
        .manager
        .findOne(Roles, {where: {role: UserRoleTypes.ADMIN}});
    const dataScientistRole = await connection
        .manager
        .findOne(Roles, {where: {role: UserRoleTypes.DATA_SCIENTIST}});
    const doctorRole = await connection
        .manager
        .findOne(Roles, {where: {role: UserRoleTypes.DOCTOR}});
    const edifierRole = await connection
        .manager
        .findOne(Roles, {where: {role: UserRoleTypes.EDIFIER}});
        
    const adminScientist = new User();
    adminScientist.email = 'admin@admin.com'.toLowerCase();
    adminScientist.login = 'admin'.toLowerCase();
    adminScientist.password = await argon2.hash('admin');
    adminScientist.roles = [adminRole, dataScientistRole, doctorRole, edifierRole];
    
    const existingUser = await connection
        .manager
        .findOne(User, {where: {login: adminScientist.login}});
        
    if (existingUser != undefined) {
        adminScientist.id = existingUser.id;
    }
    await connection.manager.save(adminScientist);
};

createSuperUser();
