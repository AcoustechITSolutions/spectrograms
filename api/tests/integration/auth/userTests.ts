import 'mocha';
import {expect} from 'chai';
import {setupServer, PORT} from '../../../src/infrastructure/server';
import {setupDatabaseConnection} from '../../../src/infrastructure/database';
import {Connection, getCustomRepository} from 'typeorm';
import supertest from 'supertest';
import express, {Application} from 'express';
import {HttpErrors, errorDescription} from '../../../src/helpers/status';
import {createTestAdmin, createTestPatient, createTestEdifier, DEFAULT_PASSWORD} from '../stubs';
import {Server} from 'http';
import axios from 'axios';
import {seedDb} from '../../../src/infrastructure/seed';
import {User} from '../../../src/infrastructure/entity/Users';
import {UserRepository} from '../../../src/infrastructure/repositories/userRepo';

let connection: Connection;
const app: Application = express();
let serverConnection: Server;

const SERVER_URL = process.env.API_URL;
let adminToken: string;
let patientToken: string;
let patient: User;
let edifier: User;
const personalData = {
    'identifier': 'test',
    'age': 20,
    'gender': 'male',
    'is_smoking': true
};

describe('Create and patch users by admin and personal data tests', function() {
    this.timeout(50000);
    beforeEach(async () => {
        return new Promise(async (resolve, reject) => {
            await setupServer(app);
            connection = await setupDatabaseConnection(false, true);
            await seedDb(connection);
            const admin = await createTestAdmin(connection);
            patient = await createTestPatient(connection);
            edifier = await createTestEdifier(connection);
            await connection.manager.save([admin, patient, edifier]);
            serverConnection = app.listen(PORT, async () => {
                console.log(`server running at port: ${PORT}`);
                const adminRes = await axios.post(`${SERVER_URL}/v1.2/admin/login`, {
                    'login': admin.login,
                    'password': DEFAULT_PASSWORD,
                });
                adminToken = adminRes.data.token;
                const patientRes = await axios.post(`${SERVER_URL}/v1.2/login`, {
                    'login': patient.login,
                    'password': DEFAULT_PASSWORD,
                });
                patientToken = patientRes.data.token;

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

    it('1.1 Can register patient with all data', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'email': 'created_patient@test.com',
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD,
                'phone_number': '1234567890',
                'comment': 'comment',
                'roles': ['patient']
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin('created_patient');
        expect(user.email).to.eq('created_patient@test.com');
        expect(user.phone_number).to.eq('1234567890');
        expect(user.comment).to.eq('comment');
        expect(user.password).to.not.be.null;
        expect(user.is_active).to.eq(true);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.2 Can register patient with minimum data', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin('created_patient');
        expect(user.password).to.not.be.null;
        expect(user.is_active).to.eq(true);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.3 Cannot register user without admin rights', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', patientToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('1.4 Cannot register user with not enough data', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });

    it('1.5 Cannot register user with wrong role', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD,
                'roles': ['badrole']
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_ROLE);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_ROLE));
    });
    
    it('1.6 Cannot register user with taken login', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': patient.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.LOGIN_TAKEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.LOGIN_TAKEN));
    });
   
    it('1.7 Cannot register patient with patients', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD,
                'patients': [patient.login]
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_PATIENTS);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_PATIENTS));
    });

    it('1.8 Can register edifier with patients', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_edifier',
                'password': DEFAULT_PASSWORD,
                'roles': ['edifier'],
                'patients': [patient.login]
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin('created_edifier');
        expect(user.password).to.not.be.null;
        expect(user.is_active).to.eq(true);
        expect(user.is_all_patients).to.eq(false);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('edifier');
        const patients = await userRepo.findPatientsByUserId(user.id);
        expect(patients.length).to.eq(1);
        expect(patients[0]).to.eq(patient.login);

        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': 'created_edifier',
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.9 Can register edifier without patients', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_edifier',
                'password': DEFAULT_PASSWORD,
                'roles': ['edifier']
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin('created_edifier');
        expect(user.password).to.not.be.null;
        expect(user.is_active).to.eq(true);
        expect(user.is_all_patients).to.eq(true);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('edifier');
        const patients = await userRepo.findPatientsByUserId(user.id);
        expect(patients.length).to.eq(0);

        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': 'created_edifier',
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('1.10 Cannot register edifier with wrong patients', async () => {
        const res = await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_edifier',
                'password': DEFAULT_PASSWORD,
                'roles': ['edifier'],
                'patients': ['badlogin']
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('2.1 Can patch user', async () => {
        await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'password': 'newpassword',
                'phone_number': '1234567890',
                'comment': 'comment',
                'is_check_healthy': true,
                'is_check_covid': false,
                'is_validate_cough': false,
                'check_start': '08:00',
                'check_end': '20:00'
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(patient.login);
        expect(user.phone_number).to.eq('1234567890');
        expect(user.comment).to.eq('comment');
        expect(user.is_check_healthy).to.eq(true);
        expect(user.is_check_covid).to.eq(false);
        expect(user.is_validate_cough).to.eq(false);
        expect(user.check_start).to.eq('08:00');
        expect(user.check_end).to.eq('20:00')

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': patient.login,
                'password': 'newpassword'
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.2 Cannot patch user without admin rights', async () => {
        const res = await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', patientToken)
            .send({
                'comment': 'comment'
            })
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('2.3 Cannot patch if user does not exist', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const maxId = await userRepo.findMaxId();
        const res = await supertest(app)
            .patch(`/v1.2/admin/user/${maxId + 1}`)
            .set('Authorization', adminToken)
            .send({
                'comment': 'comment'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('2.4 Cannot patch user with wrong checking time', async () => {
        const res1 = await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'check_start': '08:00'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res1.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res1.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));

        const res2 = await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'check_start': '80:00',
                'check_end': '20:00'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res2.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res2.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });

    it('2.5 Can patch is_active', async () => {
        await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'is_active': false 
            })
            .expect(204);

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': patient.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('2.6 Can patch user role', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(patient.login);
        expect(user.roles.length).to.eq(1);
        expect(user.roles[0].role).to.eq('patient');
        await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'roles': ['edifier']
            })
            .expect(204);

        const patchedUser = await userRepo.findByLogin(patient.login);
        expect(patchedUser.roles.length).to.eq(1);
        expect(patchedUser.roles[0].role).to.eq('edifier');

        const res = await supertest(app)
            .post('/v1.2/admin/login')
            .send({
                'login': patient.login,
                'password': DEFAULT_PASSWORD
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('2.7 Cannot patch user with wrong role', async () => {
        const res = await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'roles': ['badrole']
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_ROLE);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_ROLE));
    });
   
    it('2.8 Cannot add patients for patient', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(204);

        const res = await supertest(app)
            .patch(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .send({
                'patients': ['created_patient']
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_PATIENTS);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_PATIENTS));
    });

    it('2.9 Can add patients for edifier', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(204);

        await supertest(app)
            .patch(`/v1.2/admin/user/${edifier.id}`)
            .set('Authorization', adminToken)
            .send({
                'patients': ['created_patient']
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(edifier.login);
        expect(user.is_all_patients).to.eq(false);
        const patients = await userRepo.findPatientsByUserId(edifier.id);
        expect(patients.length).to.eq(1);
        expect(patients[0]).to.eq('created_patient');
    });

    it('2.10 Can change patients for edifier', async () => {
        await supertest(app)
            .post('/v1.2/admin/user')
            .set('Authorization', adminToken)
            .send({
                'login': 'created_patient',
                'password': DEFAULT_PASSWORD
            })
            .expect(204);

        await supertest(app)
            .patch(`/v1.2/admin/user/${edifier.id}`)
            .set('Authorization', adminToken)
            .send({
                'patients': ['created_patient']
            })
            .expect(204);

        await supertest(app)
            .patch(`/v1.2/admin/user/${edifier.id}`)
            .set('Authorization', adminToken)
            .send({
                'patients': [patient.login]
            })
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findByLogin(edifier.login);
        expect(user.is_all_patients).to.eq(false);
        const patients = await userRepo.findPatientsByUserId(edifier.id);
        expect(patients.length).to.eq(1);
        expect(patients[0]).to.eq(patient.login);
    });
    
    it('2.11 Cannot add wrong patients', async () => {
        const res = await supertest(app)
            .patch(`/v1.2/admin/user/${edifier.id}`)
            .set('Authorization', adminToken)
            .send({
                'patients': ['badlogin']
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('3.1 Cannot get zero personal data', async () => {
        const res = await supertest(app)
            .get(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .expect(404)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_PERSONAL_DATA);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_PERSONAL_DATA));
    });

    it('3.2 Can patch and get personal data', async () => {
        await supertest(app)
            .patch(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .send(personalData)
            .expect(204);

        const res = await supertest(app)
            .get(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.identifier).to.eq(personalData.identifier);
        expect(res.body.age).to.eq(personalData.age);
        expect(res.body.gender).to.eq(personalData.gender);
        expect(res.body.is_smoking).to.eq(personalData.is_smoking);
        expect(res.body.voice_audio_path).to.be.null;
    });

    it('4.1 Can change password', async () => {
        await supertest(app)
            .patch(`/v1.2/user/password`)
            .set('Authorization', patientToken)
            .send({
                'password': 'newpassword',
                'repeat_password': 'newpassword'
            })
            .expect(204);

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': patient.login,
                'password': 'newpassword'
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.token).to.not.be.null;
        expect(res.body.refreshTokenId).to.not.be.null;
    });

    it('4.2 Cannot change password with not enough data', async () => {
        const res = await supertest(app)
            .patch(`/v1.2/user/password`)
            .set('Authorization', patientToken)
            .send({
                'password': 'newpassword'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.INCORRECT_BODY);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.INCORRECT_BODY));
    });

    it('4.3 Cannot change password if repeated incorrectly', async () => {
        const res = await supertest(app)
            .patch(`/v1.2/user/password`)
            .set('Authorization', patientToken)
            .send({
                'password': 'newpassword',
                'repeat_password': 'badpassword'
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.PASSWORD_MISMATCH);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.PASSWORD_MISMATCH));
    });

    it('5.1 Can discharge personal data', async () => {
        await supertest(app)
            .patch(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .send(personalData)
            .expect(204);

        await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}/data`)
            .set('Authorization', adminToken)
            .expect(204);

        const res = await supertest(app)
            .get(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .expect(404)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_PERSONAL_DATA);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_PERSONAL_DATA));
    });

    it('5.2 Cannot discharge data without admin rights', async () => {
        await supertest(app)
            .patch(`/v1.2/user/data`)
            .set('Authorization', patientToken)
            .send(personalData)
            .expect(204);

        const res = await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}/data`)
            .set('Authorization', patientToken)
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('5.3 Cannot discharge data if user does not exist', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const maxId = await userRepo.findMaxId();
        const res = await supertest(app)
            .delete(`/v1.2/admin/user/${maxId + 1}/data`)
            .set('Authorization', adminToken)
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('5.4 Cannot discharge if no personal data', async () => {
        const res = await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}/data`)
            .set('Authorization', adminToken)
            .expect(404)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_PERSONAL_DATA);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_PERSONAL_DATA));
    });

    it('6.1 Can delete active user', async () => {
        await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .expect(204);

        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findOne(patient.id);
        expect(user.is_active).to.eq(false);
        expect(user.login).to.eq(`deleted (${patient.login})`);
        expect(user.email).to.eq(`deleted (${patient.email})`);

        const res = await supertest(app)
            .post('/v1.2/login')
            .send({
                'login': patient.login,
                'password': patient.password,
            })
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });

    it('6.2 Can delete inactive user', async () => {
        await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', adminToken)
            .expect(204);

        const res = await supertest(app)
            .post('/v1.2/user')
            .send({
                'login': process.env.TEST_EMAIL
            })
            .expect(200)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.user_id).to.not.be.null;
        const userId = res.body.user_id;
        const userRepo = getCustomRepository(UserRepository);
        const user = await userRepo.findOne(userId);
        expect(user.is_active).to.eq(false);
        expect(user.login).to.eq(process.env.TEST_EMAIL);
        expect(user.email).to.eq(process.env.TEST_EMAIL);
        expect(user.password).to.be.null;

        await supertest(app)
            .delete(`/v1.2/admin/user/${userId}`)
            .set('Authorization', adminToken)
            .expect(204);

        const deletedUser = await userRepo.findOne(userId);
        expect(deletedUser).to.be.undefined;
    });

    it('6.3 Cannot delete user without admin rights', async () => {
        const res = await supertest(app)
            .delete(`/v1.2/admin/user/${patient.id}`)
            .set('Authorization', patientToken)
            .expect(403)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.FORBIDDEN);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.FORBIDDEN));
    });

    it('6.4 Cannot delete if user does not exist', async () => {
        const userRepo = getCustomRepository(UserRepository);
        const maxId = await userRepo.findMaxId();
        const res =  await supertest(app)
            .delete(`/v1.2/admin/user/${maxId + 1}`)
            .set('Authorization', adminToken)
            .expect(400)
            .expect('Content-Type', 'application/json; charset=utf-8');

        expect(res.body.status).to.eq(HttpErrors.NO_USER);
        expect(res.body.error).to.eq(errorDescription.get(HttpErrors.NO_USER));
    });
});