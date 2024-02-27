import {User} from '../infrastructure/entity/Users';
import {createConnection} from 'typeorm';
import argon2 from 'argon2';

export const salt_passwords = async () => {
    const connection = await createConnection({
        type: 'postgres',
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        password: process.env.DB_PASSWORD,
        username: process.env.DB_USER,
        database: process.env.DB_NAME,
        logging: true,
        entities: ['dist/infrastructure/entity/*.js'],
        name: 'salt'
    });

    const users = await connection
        .manager
        .find(User);
    
    for (const user of users) {
        if (!user.password.startsWith('$argon')) {
            user.password = await argon2.hash(user.password);
            await connection.manager.save(user);
            console.log(`updated pass for ${user.login}`);
        }
    }
};
salt_passwords();