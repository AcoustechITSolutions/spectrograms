import {createConnection, Connection} from 'typeorm';

export const setupDatabaseConnection = async (isUseLogging: boolean = false, isUseTs: boolean = false): Promise<Connection> => {
    const connection = await createConnection({
        type: 'postgres',
        host: process.env.DB_HOST,
        port: Number(process.env.DB_PORT),
        password: process.env.DB_PASSWORD,
        username: process.env.DB_USER,
        database: process.env.DB_NAME,
        logging: isUseLogging,
        entities: isUseTs ? ['src/infrastructure/entity/*.ts'] 
            : ['dist/infrastructure/entity/*.js'],
        migrations: isUseTs ? ['src/infrastructure/migrations/*.ts']
            : ['dist/infrastructure/migrations/*.js'],
    });
    await connection.runMigrations();
    return connection;
};
