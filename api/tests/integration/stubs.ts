import {User} from '../../src/infrastructure/entity/Users';
import {Roles} from '../../src/infrastructure/entity/Roles';
import {UserRoleTypes} from '../../src/domain/UserRoles';
import {Connection} from 'typeorm';
import argon2 from 'argon2';

export const DEFAULT_PASSWORD = 'test';
export const createTestPatient = async (connection: Connection) => {
    const userRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.PATIENT}});

    const testUser = new User();
    testUser.login = 'test_patient';
    testUser.password = await argon2.hash(DEFAULT_PASSWORD);
    testUser.email = process.env.TEST_EMAIL;
    testUser.roles = [userRole];

    return testUser;
};

export const createTestDataset = async (connection: Connection) => {
    const datasetRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DATASET}});

    const testDataset = new User();
    testDataset.login = 'test_dataset';
    testDataset.password = await argon2.hash(DEFAULT_PASSWORD);
    testDataset.roles = [datasetRole];

    return testDataset;
};

export const createTestAdmin = async (connection: Connection) => {
    const adminRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.ADMIN}});

    const testAdmin = new User();
    testAdmin.login = 'test_admin';
    testAdmin.password = await argon2.hash(DEFAULT_PASSWORD);
    testAdmin.roles = [adminRole];

    return testAdmin;
};

export const createTestEdifier = async (connection: Connection) => {
    const edifierRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.EDIFIER}});

    const testEdifier = new User();
    testEdifier.login = 'test_edifier';
    testEdifier.password = await argon2.hash(DEFAULT_PASSWORD);
    testEdifier.roles = [edifierRole];

    return testEdifier;
};

export const createTestViewer = async (connection: Connection) => {
    const viewerRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.VIEWER}});

    const testViewer = new User();
    testViewer.login = 'test_viewer';
    testViewer.password = await argon2.hash(DEFAULT_PASSWORD);
    testViewer.roles = [viewerRole];

    return testViewer;
};

export const createTestDoctor = async (connection: Connection) => {
    const doctorRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DOCTOR}});

    const testDoctor = new User();
    testDoctor.login = 'test_doctor';
    testDoctor.password = await argon2.hash(DEFAULT_PASSWORD);
    testDoctor.roles = [doctorRole];

    return testDoctor;
};

export const createTestDataScientist = async (connection: Connection) => {
    const datascientistRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DATA_SCIENTIST}});

    const testDataScientist = new User();
    testDataScientist.login = 'test_datascientist';
    testDataScientist.password = await argon2.hash(DEFAULT_PASSWORD);
    testDataScientist.roles = [datascientistRole];

    return testDataScientist;
};

export const createTestSuperuser = async (connection: Connection) => {
    const adminRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.ADMIN}});
    const edifierRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.EDIFIER}});
    const viewerRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.VIEWER}});
    const doctorRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DOCTOR}});
    const datascientistRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DATA_SCIENTIST}});
    const patientRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.PATIENT}});
    const datasetRole = await connection.manager.findOneOrFail(Roles, {where: {role: UserRoleTypes.DATASET}});

    const testSuperuser = new User();
    testSuperuser.login = 'test_superuser';
    testSuperuser.password = await argon2.hash(DEFAULT_PASSWORD);
    testSuperuser.roles = [adminRole, edifierRole, viewerRole, doctorRole, datascientistRole, patientRole, datasetRole];

    return testSuperuser;
};
