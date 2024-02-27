import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';

import {createTestDataset, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import axios from 'axios';
import {join} from 'path';
import {seedDb} from '../../../src/infrastructure/seed';

const app: Application = express();
let connection: Connection;
let serverConnection: Server;
let token: string;
const SERVER_URL = process.env.API_URL;
const MIN_REQUEST_ID = 1;

// describe('Dataset tests', function() {
//     this.timeout(50000);
//     beforeEach(async () => {
//         return new Promise(async (resolve, reject) => {
//             await setupServer(app);
//             connection = await setupDatabaseConnection(false, true);
//             await seedDb(connection);
//             const user = await createTestUser(connection);
//             await connection.manager.save(user);
//             serverConnection = app.listen(PORT, async () => {
//                 console.log(`server running at port: ${PORT}`);
//                 const res = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': user.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 token = res.data.token;
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

//     it('Should create diagnostic with acute bronchitis and return from get results', async () => {
//         const reqData = {
//             age: 24,
//             gender: 'male',
//             isSmoking: true,
//             isForce: false,
//             sickDays: 21,
//             identifier: '1234',
//             diseaseType: 'acute',
//             disease: 'acute_bronchitis',
//             privacyEulaVersion: 1,
//         };
//         const createResponse = await supertest(app)
//             .post('/v1/dataset')
//             .set('Authorization', token)
//             .field('age', reqData.age)
//             .field('gender', reqData.gender)
//             .field('is_smoking', reqData.isSmoking)
//             .field('sick_days', reqData.sickDays)
//             .field('is_force', reqData.isForce)
//             .field('identifier', reqData.identifier)
//             .field('disease_type', reqData.diseaseType)
//             .field('disease', reqData.disease)
//             .field('privacy_eula_version', reqData.privacyEulaVersion)
//             .attach('cough_audio', join(__dirname, '../testFiles/testAudio/testCough.wav'))
//             .attach('speech_audio', join(__dirname, '../testFiles/testAudio/testSpeech.wav'))
//             .attach('breath_audio', join(__dirname, '../testFiles/testAudio/testBreath.wav'))
//             .expect(201)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(createResponse.body.request_id)
//             .to.not.be.null;
//         expect(createResponse.body.request_id)
//             .to.be.a('number')
//             .and.satisfy(Number.isInteger)
//             .and.at.least(MIN_REQUEST_ID);
//     });

//     it('Should not create acute disease with null type', async () => {
//         const reqData = {
//             age: 24,
//             gender: 'male',
//             isSmoking: true,
//             isForce: false,
//             sickDays: 21,
//             identifier: '1234',
//             diseaseType: 'acute',
//             privacyEulaVersion: 1,
//         };
//         const createResponse = await supertest(app)
//             .post('/v1/dataset')
//             .set('Authorization', token)
//             .field('age', reqData.age)
//             .field('gender', reqData.gender)
//             .field('is_smoking', reqData.isSmoking)
//             .field('sick_days', reqData.sickDays)
//             .field('is_force', reqData.isForce)
//             .field('identifier', reqData.identifier)
//             .field('disease_type', reqData.diseaseType)
//             .field('privacy_eula_version', reqData.privacyEulaVersion)
//             .attach('cough_audio', join(__dirname, '../testFiles/testAudio/testCough.wav'))
//             .attach('speech_audio', join(__dirname, '../testFiles/testAudio/testSpeech.wav'))
//             .attach('breath_audio', join(__dirname, '../testFiles/testAudio/testBreath.wav'))
//             .expect(400)
//             .expect('Content-Type', 'application/json; charset=utf-8');
//     });
// });
