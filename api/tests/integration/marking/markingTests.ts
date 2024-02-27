import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';

import {createTestDataset, createTestEdifier, createTestDoctor, createTestDataScientist, createTestSuperuser, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import axios from 'axios';

import {seedDb} from '../../../src/infrastructure/seed';

const app: Application = express();
let connection: Connection;
let serverConnection: Server;

const SERVER_URL = process.env.API_URL;
const MIN_REQUEST_ID = 1;

// describe('Check marking route user permission', function() {
//     this.timeout(50000);
//     let dataScientistToken: string;
//     let userToken: string;
//     let adminToken: string;
//     let adminDataScientistToken: string;

//     beforeEach(async () => {
//         return new Promise(async (resolve) => {
//             await setupServer(app);
//             connection = await setupDatabaseConnection(false, true);
//             await seedDb(connection);
//             const user = await createTestUser(connection);
//             const admin = await createTestAdmin(connection);
//             const dataScientist = await createTestDataScientist(connection);
//             const adminDataScientist = await createTestAdminDataScientist(connection);
//             await connection.manager.save([user, admin, dataScientist, adminDataScientist]);

//             serverConnection = app.listen(PORT, async () => {
//                 console.log(`server running at port: ${PORT}`);
//                 const scientistToken = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': dataScientist.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 dataScientistToken = scientistToken.data.token;
//                 const userRes = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': user.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 userToken = userRes.data.token;
//                 const adminRes = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': admin.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 adminToken = adminRes.data.token;
//                 const adminDataScientistRes = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': adminDataScientist.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 adminDataScientistToken = adminDataScientistRes.data.token;
//                 resolve();
//             });
//         });
//     });

//     afterEach(async () => {
//         return new Promise(async (resolve, reject) => {
//             await connection.dropDatabase();
//             await connection.close();
//             await serverConnection.close(reject);
//             resolve();
//         });
//     });

//     it('Shoud return 403 for usual user', async () => {
//         const result = await supertest(app)
//             .get('/v1/admin/marking')
//             .set('Authorization', userToken)
//             .expect(403)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(result.body.status).to.be.eq('error');
//         expect(result.body.error).to.be.eq('User does not have permission to view this resource');
//     });

//     it('Should return 403 for admin', async () => {
//         const result = await supertest(app)
//             .get('/v1/admin/marking')
//             .set('Authorization', adminToken)
//             .expect(403)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(result.body.status).to.be.eq('error');
//         expect(result.body.error).to.be.eq('User does not have permission to view this resource');
//     });
// });
