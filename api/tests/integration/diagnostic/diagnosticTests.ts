import 'mocha';

import {Connection} from 'typeorm';

import express, {Application} from 'express';

import {Server} from 'http';

const app: Application = express();
let connection: Connection;
let serverConnection: Server;
let token: string;
let adminToken: string;
const SERVER_URL = process.env.API_URL;
const MIN_REQUEST_ID = 1;
// FIXME: fix ml service audio folder permissions
// const createDiagnosticAndCheck = async () => {
//     const createResponse = await supertest(app)
//         .post('/v1/diagnostic')
//         .set('Authorization', token)
//         .field('age', '23')
//         .field('gender', 'male')
//         .field('is_smoking', 'false')
//         .field('duration', '0')
//         .field('is_force', 'false')
//         .attach('cough_audio', join(__dirname, '../testFiles/testAudio/testCough.wav'))
//         .expect(201)
//         .expect('Content-Type', 'application/json; charset=utf-8');

//     expect(createResponse.body).to.be.not.null;
//     expect(createResponse.body.request_id).to.be.not.null;
//     expect(createResponse.body.request_id)
//         .to.be.a('number')
//         .and.satisfy(Number.isInteger)
//         .and.at.least(MIN_REQUEST_ID);
//     return createResponse;
// };

// const loadDiagnosticFromBDExpectProcessingStatus = async (requestId: number) => {
//     const diagnosticRequest = await connection.manager.findOneOrFail(DiagnosticRequest, {
//         relations: ['status', 'diagnostic_report', 'cough_audio', 'patient_info', 'cough_characteristics'],
//         where: {
//             id: requestId,
//         },
//     });

//     const processingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//         where: {
//             request_status: DomainDiagnosticRequestStatus.PROCESSING,
//         },
//     });

//     expect(diagnosticRequest.status).to.be.eql(processingStatus);
//     return diagnosticRequest;
// };

// const getDiagnosticExpectPending = async (requestId: number) => {
//     const response = await supertest(app)
//         .get('/v1/diagnostic')
//         .set('Authorization', token)
//         .expect(200)
//         .expect('Content-Type', 'application/json; charset=utf-8');

//     expect(response.body).to.not.be.null;
//     expect(response.body.length).to.be.eq(1);
//     expect(response.body[0].status).to.be.eq('pending');
//     expect(response.body[0].request_id).to.be.eq(requestId);

//     return response;
// };

// describe('Diagnostic tests with ml service', function() {
//     this.timeout(30000)
//     beforeEach(async () => {
//         return new Promise(async (resolve, reject) => {
//             await setupServer(app);
//             connection = await setupDatabaseConnection(false, true);
//             await seedDb(connection);
//             const testUser = await createTestUser(connection);
//             const testAdmin = await createTestAdmin(connection);
//             await connection.manager.save([testUser, testAdmin]);
//             serverConnection = app.listen(PORT, async () => {
//                 console.log(`server running at port: ${PORT}`);
//                 const res = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': testUser.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 token = res.data.token;
//                 const adminRes = await axios.post(`${SERVER_URL}/v1/login`, {
//                     'login': testAdmin.login,
//                     'password': DEFAULT_PASSWORD,
//                 });
//                 adminToken = adminRes.data.token;
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

//     it('Check api return for get /diagnostic', async () => {
//         const createResponse = await createDiagnosticAndCheck();

//         const requestId = createResponse.body.request_id;
//         let getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(0);

//         const request = await connection.manager.findOneOrFail(DiagnosticRequest, {
//             relations: ['status', 'cough_characteristics', 'diagnostic_report', 'cough_audio',
//                 'diagnostic_report.diagnosis', 'cough_characteristics.productivity', 'cough_characteristics.intensity'],
//             where: {id: requestId},
//         });
//         const errorStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.ERROR},
//         });
//         request.status = errorStatus;
//         await connection.manager.save(request);

//         getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.ERROR,
//             date: request.dateCreated.toISOString(),
//             request_id: requestId,
//         });

//         const pendingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.PENDING},
//         });
//         request.status = pendingStatus;
//         await connection.manager.save(request);

//         getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.PENDING,
//             date: request.dateCreated.toISOString(),
//             request_id: requestId,
//         });

//         const successStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {request_status: DomainDiagnosticRequestStatus.SUCCESS},
//         });
//         request.status = successStatus;
//         request.diagnostic_report.diagnosis_probability = 0.33;

//         const covidType = await connection.manager.findOneOrFail(DiagnosisTypesEntity, {
//             where: {diagnosis_type: DiagnosisTypes.COVID_19},
//         });

//         const productiveCough = await connection.manager.findOneOrFail(CoughProductivityTypes, {
//             where: {productivity_type: CoughProductivity.PRODUCTIVE},
//         });
//         const paroxysmalCough = await connection.manager.findOneOrFail(CoughIntensityTypes, {
//             where: {intensity_type: CoughIntensity.PAROXYSMAL},
//         });

//         request.diagnostic_report.diagnosis = covidType;
//         await connection.manager.save([request, request.diagnostic_report]);

//         getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.SUCCESS,
//             date: request.dateCreated.toISOString(),
//             request_id: requestId,
//             report: {
//                 diagnosis: DiagnosisTypes.COVID_19,
//                 probability: 0.33,
//             },
//         });

//         request.diagnostic_report.commentary = 'Good';
//         await connection.manager.save([request.diagnostic_report]);

//         getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.SUCCESS,
//             date: request.dateCreated.toISOString(),
//             request_id: requestId,
//             report: {
//                 commentary: 'Good',
//                 diagnosis: DiagnosisTypes.COVID_19,
//                 probability: 0.33,
//             },
//         });

//         request.cough_characteristics.intensity = paroxysmalCough;
//         request.cough_characteristics.productivity = productiveCough;
//         await connection.manager.save([request.cough_characteristics]);

//         getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.SUCCESS,
//             date: request.dateCreated.toISOString(),
//             request_id: requestId,
//             report: {
//                 commentary: 'Good',
//                 diagnosis: DiagnosisTypes.COVID_19,
//                 probability: 0.33,
//                 intensity: CoughIntensity.PAROXYSMAL,
//                 productivity: CoughProductivity.PRODUCTIVE,
//             },
//         });
//     });

//     it('Should create diagnostic and return from get results', async () => {
//         const createResponse = await createDiagnosticAndCheck();

//         let diagnosticRequest = await connection.manager.findOneOrFail(DiagnosticRequest, {
//             relations: ['status'],
//             where: {
//                 id: createResponse.body.request_id,
//             },
//         });

//         const pendingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {
//                 request_status: DomainDiagnosticRequestStatus.PENDING,
//             },
//         });
//         const processingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {
//                 request_status: DomainDiagnosticRequestStatus.PROCESSING,
//             },
//         });

//         expect(diagnosticRequest.status).to.be.eql(processingStatus);
//         await delay(20000);

//         diagnosticRequest = await connection.manager.findOneOrFail(DiagnosticRequest, {
//             relations: ['status', 'diagnostic_report', 'cough_audio'],
//             where: {
//                 id: createResponse.body.request_id,
//             },
//         });
//         expect(diagnosticRequest.status).to.be.eql(pendingStatus);

//         await getDiagnosticExpectPending(createResponse.body.request_id);

//         expect(diagnosticRequest.diagnostic_report.diagnosis_probability)
//             .to.be.greaterThan(0)
//             .and.lessThan(1)
//             .and.not.null;

//         expect(diagnosticRequest.cough_audio.samplerate).to.be.eq(44100);
//     });

//     it('Check doctor workflow with diagnostic', async () => {
//         const createResponse = await createDiagnosticAndCheck();

//         let diagnosticRequest = await loadDiagnosticFromBDExpectProcessingStatus(createResponse.body.request_id);

//         const pendingStatus = await connection.manager.findOneOrFail(DiagnosticRequestStatus, {
//             where: {
//                 request_status: DomainDiagnosticRequestStatus.PENDING,
//             },
//         });

//         await delay(20000);

//         await getDiagnosticExpectPending(diagnosticRequest.id);

//         diagnosticRequest = await connection.manager.findOneOrFail(DiagnosticRequest, {
//             relations: ['status', 'diagnostic_report', 'cough_audio', 'patient_info', 'cough_characteristics'],
//             where: {
//                 id: createResponse.body.request_id,
//             },
//         });
//         expect(diagnosticRequest.diagnostic_report.diagnosis_probability)
//             .to.be.greaterThan(0)
//             .and.lessThan(1)
//             .and.not.null;

//         expect(diagnosticRequest.cough_audio.samplerate).to.be.eq(44100);

//         const requestId = diagnosticRequest.id;

//         const TEST_PROBABILITY = 0.000001;
//         const TEST_DIAGNOSIS = DiagnosisTypes.HEALTHY;
//         const TEST_COMMENTARY = 'Dobje';
//         const TEST_PRODUCTIVE = CoughProductivity.PRODUCTIVE;
//         const TEST_INTENSITY = CoughIntensity.PAROXYSMAL;

//         await supertest(app)
//             .patch(`/v1/admin/processing/${requestId}`)
//             .set('Authorization', adminToken)
//             .send({
//                 'diagnosis_probability': TEST_PROBABILITY,
//                 'diagnosis': TEST_DIAGNOSIS,
//                 'productivity': TEST_PRODUCTIVE,
//                 'intensity': TEST_INTENSITY,
//                 'commentary': TEST_COMMENTARY,
//             })
//             .expect(200);

//         const getResponse = await supertest(app)
//             .get('/v1/diagnostic')
//             .set('Authorization', token)
//             .expect(200)
//             .expect('Content-Type', 'application/json; charset=utf-8');

//         expect(getResponse.body.length).to.be.eq(1);
//         expect(getResponse.body[0]).to.be.eql({
//             status: DomainDiagnosticRequestStatus.SUCCESS,
//             date: diagnosticRequest.dateCreated.toISOString(),
//             request_id: requestId,
//             report: {
//                 commentary: TEST_COMMENTARY,
//                 diagnosis: TEST_DIAGNOSIS,
//                 probability: TEST_PROBABILITY,
//                 intensity: TEST_INTENSITY,
//                 productivity: TEST_PRODUCTIVE,
//             },
//         });
//     });
// });
