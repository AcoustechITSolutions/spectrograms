import {setupDatabaseConnection} from './infrastructure/database';
import {seedDb} from './infrastructure/seed';
import {setupServer, PORT} from './infrastructure/server';
import express, {Application} from 'express';

const app: Application = express();

const start = async () => {
    const connection = await setupDatabaseConnection(process.env.NODE_ENV != 'production', false);
    await seedDb(connection);
    await setupServer(app);
    app.listen(PORT, () => {
        console.log(`server running at port: ${PORT}`);
    });
};

start();
export {
    app,
};
