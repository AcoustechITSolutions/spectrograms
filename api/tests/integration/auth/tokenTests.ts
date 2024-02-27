import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';
import {HttpErrors, errorDescription} from '../../../src/helpers/status';
import {createTestPatient, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import axios from 'axios';
import {seedDb} from '../../../src/infrastructure/seed';
import {delay} from '../helper';
import config from '../../../src/config/config';
import {User} from '../../../src/infrastructure/entity/Users';

let connection: Connection;
const app: Application = express();
let serverConnection: Server;

const SERVER_URL = process.env.API_URL;
let patient: User;
let token: string;
let refreshToken: string;

describe('Token refresh and logout tests', function() {
    this.timeout(100000);
    beforeEach(async () => {
        return new Promise(async (resolve, reject) => {
            await setupServer(app);
            connection = await setupDatabaseConnection(false, true);
            await seedDb(connection);
            patient = await createTestPatient(connection);
            await connection.manager.save(patient);
            serverConnection = app.listen(PORT, async () => {
                console.log(`server running at port: ${PORT}`);
                const res = await axios.post(`${SERVER_URL}/v1.2/login`, {
                    'login': patient.login,
                    'password': DEFAULT_PASSWORD
                });
                token = res.data.token;
                refreshToken = res.data.refreshTokenId;
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

    it('1.1 Token expires with time', async () => {
        const firstResponse = await supertest(app)
            .get('/v1.2/diagnostic')
            .set('Authorization', token)
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(firstResponse.body).to.eql([]);
        await delay(config.jwtLife);

        const secondResponse = await supertest(app)
            .get('/v1.2/diagnostic')
            .set('Authorization', token)
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(secondResponse.body.status).to.eq(HttpErrors.TOKEN_VERIFICATION_ERROR);
        expect(secondResponse.body.error).to.eq(errorDescription.get(HttpErrors.TOKEN_VERIFICATION_ERROR));
    });

    it('1.2 Can refresh token', async () => {
        const firstResponse = await supertest(app)
            .get('/v1.2/diagnostic')
            .set('Authorization', token)
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(firstResponse.body).to.eql([]);
        await delay(config.jwtLife);

        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;

        const secondResponse = await supertest(app)
            .get('/v1.2/diagnostic')
            .set('Authorization', res.body.token)
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(secondResponse.body).to.eql([]);
    });

    it('1.3 Cannot refresh without token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_TOKEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_TOKEN));
    });

    it('1.4 Cannot refresh without refresh token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_REFRESH_TOKEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_REFRESH_TOKEN));
    });

    it('1.5 Cannot refresh with wrong token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', 'badtoken')
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.TOKEN_VERIFICATION_ERROR));
    });

    it('1.6 Cannot refresh with wrong refresh token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .send({
                'refreshTokenId': 'badrefresh'
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR));
    });
   
    it('1.7 Cannot refresh if refresh token expired', async () => {
        await delay(config.jwtRefreshLife);

        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR));
    });

    it('1.8 Cannot refresh if user is not active', async () => {
        patient.is_active = false;
        await connection.manager.save(patient);

        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('2.1 Can logout', async () => {
        await supertest(app)
            .post('/v1.2/token/logout')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(204);

        const res = await supertest(app)
            .post('/v1.2/token/refresh')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR));
    });

    it('2.2 Cannot logout with wrong token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/logout')
            .set('Authorization', 'badtoken')
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.TOKEN_VERIFICATION_ERROR));
    });

    it('2.3 Cannot logout without token', async () => {
        const res = await supertest(app)
            .post('/v1.2/token/logout')
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_TOKEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_TOKEN));
    });

    it('2.4 Cannot logout if already logged out', async () => {
        await supertest(app)
            .post('/v1.2/token/logout')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(204);

        const res = await supertest(app)
            .post('/v1.2/token/logout')
            .set('Authorization', token)
            .send({
                'refreshTokenId': refreshToken
            })
            .expect(401)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.REFRESH_TOKEN_VERIFICATION_ERROR));
    });
});