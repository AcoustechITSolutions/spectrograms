import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';
import {HttpErrors, errorDescription} from '../../../src/helpers/status';
import {createTestPatient, createTestDataset, createTestAdmin, createTestEdifier, createTestViewer, createTestDoctor,
    createTestDataScientist, createTestSuperuser, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import {seedDb} from '../../../src/infrastructure/seed';

let connection: Connection;
const app: Application = express();
let serverConnection: Server;

describe('Authorization tests', function() {
    this.timeout(50000);
    beforeEach(async () => {
        return new Promise(async (resolve, reject) => {
            await setupServer(app);
            connection = await setupDatabaseConnection(false, true);
            await seedDb(connection);
            serverConnection = app.listen(PORT, () => {
                console.log(`server running at port: ${PORT}`);
                resolve();
            });
        });
    });

    afterEach(async () => {
        return new Promise(async (resolve, reject) => {
            await connection.dropDatabase();
            await connection.close();
            await serverConnection.close(reject);
            resolve();
        });
    });

    it('1.1 Cannot login if user does not exist', async () => {
        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': 'badlogin',
                'password': 'badpassword',
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('1.2 Can login as patient', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': user.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.3 Can login as dataset user', async () => {
        const dataset = await createTestDataset(connection);
        await connection.manager.save(dataset);
        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': dataset.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.4 Cannot login as patient with wrong password', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': user.login,
                'password': 'badpassword'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.PASSWORD_MISMATCH);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.PASSWORD_MISMATCH));
    });

    it('1.5 Cannot login as patient without password', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': user.login
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });

    it('2.1 Cannot login as patient at /admin/login', async () => {
        const user = await createTestPatient(connection);
        await connection.manager.save(user);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': user.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('2.2 Cannot login as dataset user at /admin/login', async () => {
        const dataset = await createTestDataset(connection);
        await connection.manager.save(dataset);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': dataset.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('2.3 Can login as admin at /admin/login', async () => {
        const admin = await createTestAdmin(connection);
        await connection.manager.save(admin);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': admin.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.4 Can login as edifier at /admin/login', async () => {
        const edifier = await createTestEdifier(connection);
        await connection.manager.save(edifier);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': edifier.login,
                'password': DEFAULT_PASSWORD,
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.5 Can login as viewer at /admin/login', async () => {
        const viewer = await createTestViewer(connection);
        await connection.manager.save(viewer);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': viewer.login,
                'password': DEFAULT_PASSWORD,
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.6 Can login as doctor at /admin/login', async () => {
        const doctor = await createTestDoctor(connection);
        await connection.manager.save(doctor);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': doctor.login,
                'password': DEFAULT_PASSWORD,
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.7 Can login as datascientist at /admin/login', async () => {
        const dataScientist = await createTestDataScientist(connection);
        await connection.manager.save(dataScientist);
        const res = await supertest(app)
            .post('/v1/admin/login')
            .send({
                'login': dataScientist.login,
                'password': DEFAULT_PASSWORD,
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.8 Can login as superuser at /admin/login', async () => {
        const superuser = await createTestSuperuser(connection);
        await connection.manager.save(superuser);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': superuser.login,
                'password': DEFAULT_PASSWORD,
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.9 Cannot login at /admin/login with wrong password', async () => {
        const admin = await createTestAdmin(connection);
        await connection.manager.save(admin);
        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': admin.login,
                'password': 'badpassword'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.PASSWORD_MISMATCH);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.PASSWORD_MISMATCH));
    });

    it('2.10 Cannot login at /admin/login without password', async () => {
        const admin = await createTestAdmin(connection);
        await connection.manager.save(admin);
        let res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': admin.login
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });
});
