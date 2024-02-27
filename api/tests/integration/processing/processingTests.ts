import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';
import {createTestPatient, createTestEdifier, createTestDoctor, createTestDataScientist, createTestSuperuser, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import axios from 'axios';
import {join} from 'path';
import {seedDb} from '../../../src/infrastructure/seed';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus} from '../../../src/domain/RequestStatus';
import {DiagnosticRequestStatus} from '../../../src/infrastructure/entity/DiagnostRequestStatus';
import {DiagnosticRequest as EntityDiagnosticRequest} from '../../../src/infrastructure/entity/DiagnosticRequest';

const app: Application = express();
let connection: Connection;
let serverConnection: Server;

const SERVER_URL = process.env.API_URL;
const MIN_REQUEST_ID = 1;

// describe('Processing tests', function() {
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
//             .get('/v1/admin/processing')
//             .set('Authorization', userToken)
//             .expect(403)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(result.body.status).to.be.eq('error');
//         expect(result.body.error).to.be.eq('User does not have permission to view this resource');
//     });

//     it('Should return 200 for doctor', async () => {
//         const result = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');
//     });

//     it('Should return 200 for admin data scientist', async () => {
//         const result = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminDataScientistToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');
//     });

//     it('Should return 403 for data scientist', async () => {
//         const result = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', dataScientistToken)
//             .expect(403)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(result.body.status).to.be.eq('error');
//         expect(result.body.error).to.be.eq('User does not have permission to view this resource');
//     });

//     it('Test valid get api return', async () => {
//         const reqData = {
//             age: 24,
//             gender: 'male',
//             isSmoking: true,
//             isForce: false,
//             sickDays: 21,
//             identifier: 'testCough',
//             diseaseType: 'acute',
//             disease: 'acute_bronchitis',
//             privacyEulaVersion: 1,
//         };
//         const createResponse = await supertest(app)
//             .post('/v1/diagnostic')
//             .set('Authorization', userToken)
//             .field('age', reqData.age)
//             .field('gender', reqData.gender)
//             .field('is_smoking', reqData.isSmoking)
//             .field('sick_days', reqData.sickDays)
//             .field('is_force', reqData.isForce)
//             .attach('cough_audio', join(__dirname, '../testFiles/testAudio/testCough.wav'))
//             .expect(201)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         const requestId = createResponse.body.request_id;

//         let getResponse = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body).to.be.not.null;
//         expect(getResponse.body.length).to.be.eq(0);

//         const request = await connection.manager.findOneOrFail(EntityDiagnosticRequest, {
//             relations: ['status', 'cough_audio'],
//             where: {
//                 id: requestId,
//             },
//         });

//         const pendingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.PENDING},
//         });
//         const errorStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.ERROR},
//         });
//         const successStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.SUCCESS},
//         });
//         const processingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.PROCESSING},
//         });

//         request.status = pendingStatus;
//         await connection.manager.save(request);
//         const req = await connection.manager.findOneOrFail(EntityDiagnosticRequest, {
//             relations: ['status'],
//             where: {
//                 id: requestId,
//             },
//         });

//         getResponse = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body).to.be.not.null;
//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.PENDING,
//             identifier: reqData.identifier,
//             age: reqData.age,
//             request_id: requestId,
//             gender: reqData.gender,
//             date_created: request.dateCreated.toISOString(),
//         });

//         request.status = errorStatus;
//         await connection.manager.save(request);

//         getResponse = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body).to.be.not.null;
//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.ERROR,
//             identifier: reqData.identifier,
//             age: reqData.age,
//             request_id: requestId,
//             gender: reqData.gender,
//             date_created: request.dateCreated.toISOString(),
//         });

//         request.status = successStatus;
//         await connection.manager.save(request);

//         getResponse = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body).to.be.not.null;
//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.SUCCESS,
//             identifier: reqData.identifier,
//             age: reqData.age,
//             request_id: requestId,
//             gender: reqData.gender,
//             date_created: request.dateCreated.toISOString(),
//         });

//         request.status = processingStatus;
//         await connection.manager.save(request);

//         getResponse = await supertest(app)
//             .get('/v1/admin/processing')
//             .set('Authorization', adminToken)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body).to.be.not.null;
//         expect(getResponse.body.length).to.be.eq(0);
//     });
// });
