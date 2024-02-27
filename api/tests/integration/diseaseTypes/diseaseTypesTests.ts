import 'mocha';
import {expect} from 'chai';
import supertest from 'supertest';
import {createTestDataset, DEFAULT_PASSWORD} from '../stubs';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection} from 'typeorm';
import express, {Application} from 'express';
import {Server} from 'http';
import axios from 'axios';

const app: Application = express();
let connection: Connection;
let serverConnection: Server;
let token: string;
const SERVER_URL = process.env.API_URL;

const EXPECTED_DISEASES = [
    'acute',
    'chronic',
    'none',
];

const EXPECTED_EN_CHRONIC_DISEASES = [
    {'key': 'other', 'value': 'Other'},
    {'key': 'lung_infarction', 'value': 'Lung infarction'},
    {'key': 'bronchial_asthma', 'value': 'Bronchial asthma'},
    {'key': 'psychogenic_cough', 'value': 'Psychogenic cough'},
    {'key': 'primary_tuberculosis_complex', 'value': 'Primary tuberculosis complex'},
    {'key': 'chronical_bronchitis', 'value': 'Chronic bronchitis'},
    {'key': 'copd', 'value': 'COPD'},
    {'key': 'bronchoectatic_disease', 'value': 'Bronchiectatic disease'},
    {'key': 'tumors', 'value': 'Tumors'},
    {'key': 'congestive_heart_failure', 'value': 'Congestive heart failure'},
];

const EXPECTED_EN_ACUTE_DISEASES = [
    {'key': 'other', 'value': 'Other'},
    {'key': 'acute_bronchitis', 'value': 'Acute bronchitis'},
    {'key': 'viral_pneumonia', 'value': 'Viral pneumonia'},
    {'key': 'pleurisy', 'value': 'Pleurisy'},
    {'key': 'pulmonary_embolism', 'value': 'Pulmonary embolism'},
    {'key': 'whooping_cough', 'value': 'Pertussis'},
    {'key': 'pneumonia', 'value': 'Pneumonia'},
    {'key': 'pneumonia_complication', 'value': 'Pneumonia complication'},
];

// describe('Check disease types', function() {
//     this.timeout(50000);
//     beforeEach(async () => {
//         return new Promise(async (resolve, reject) => {
//             await setupServer(app);
//             connection = await setupDatabaseConnection(false, true);
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

//     it('Disease types are not allowed for an unathorized user', async () => {
//         await supertest(app)
//             .get('/v1/disease_types')
//             .expect(401)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('Token not provided');
//             });
//     });

//     it('Should be token verification error', async () => {
//         await supertest(app)
//             .get('/v1/disease_types')
//             .set('Authorization', 'BAD TOKEN')
//             .expect(401)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('Token verification error');
//             });
//     });

//     it('Should return disesase types', async () => {
//         await supertest(app)
//             .get('/v1/disease_types')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 console.log(response.body);
//                 expect(response.body).to.be.eql(EXPECTED_DISEASES);
//             });
//     });

//     it('Should return english chronic types', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/chronic')
//             .query({lang: 'en'})
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body).to.be.eql(EXPECTED_EN_CHRONIC_DISEASES);
//             });
//     });

//     it('Chronic types error without lang param', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/chronic')
//             .set('Authorization', token)
//             .expect(400)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('No such language');
//             });
//     });

//     it('Chronic types should fail without valid token', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/chronic')
//             .set('Authorization', 'BAD TOKEN')
//             .expect(401)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('Token verification error');
//             });
//     });

//     it('Acute types should fail without valid token', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/acute')
//             .set('Authorization', 'BAD')
//             .expect(401)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('Token verification error');
//             });
//     });

//     it('Acute types error without lang param', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/acute')
//             .set('Authorization', token)
//             .expect(400)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body.status).to.be.eq('error');
//                 expect(response.body.error).to.be.eq('No such language');
//             });
//     });

//     it('Should return english acute diseases', async () => {
//         await supertest(app)
//             .get('/v1/disease_types/acute')
//             .set('Authorization', token)
//             .query({lang: 'en'})
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8')
//             .then((response) => {
//                 expect(response.body).to.be.eql(EXPECTED_EN_ACUTE_DISEASES);
//             });
//     });
// });
