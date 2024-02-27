import {Connection, ObjectType, EntitySchema} from 'typeorm';
import {AcuteCoughTypes as DomainAcuteCoughTypes, ChronicCoughTypes as DomainChronicCoughTypes, DiseaseTypes as DomainDiseaseTypes} from '../domain/DiseaseTypes';
import {AcuteCoughTypes as EntityAcuteCoughTypes} from '../infrastructure/entity/AcuteCoughTypes';
import {ChronicCoughTypes as EntityChornicCoughTypes} from '../infrastructure/entity/ChronicCoughTypes';
import {DiseaseTypes as EntityDiseaseTypes} from '../infrastructure/entity/DiseaseTypes';
import {Roles} from '../infrastructure/entity/Roles';
import {UserRoleTypes} from '../domain/UserRoles';
import {User} from '../infrastructure/entity/Users';
import {CoughProductivityTypes as EntityCoughProductivityTypes} from '../infrastructure/entity/CoughProductivityTypes';
import {CoughProductivity as DomainCoughProductivityTypes, CoughIntensity as DomainCoughIntensityTypes} from '../domain/CoughTypes';
import {DiagnosisTypes as DomainDiagnosisTypes} from '../domain/DiagnosisTypes';
import {DiagnosisTypes as EntityDiagnosisTypes} from '../infrastructure/entity/DiagnosisTypes';
import {DiagnosticRequestStatus as EntityDiagnosticRequestStatus} from '../infrastructure/entity/DiagnostRequestStatus';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus,
    HWRequestStatus as DomainHWRequestStatus,
    DatasetRequestStatus as DomainDatasetRequestStatus,
} from '../domain/RequestStatus';
import {CoughIntensityTypes as EntityCoughIntensityTypes} from '../infrastructure/entity/CoughIntensityTypes';
import {QueryDeepPartialEntity} from 'typeorm/query-builder/QueryPartialEntity';
import {BreathingTypes as BreathingTypesEntity} from '../infrastructure/entity/BreathingTypes';
import {BreathingDepthTypes as BreathingDepthTypesEntity} from '../infrastructure/entity/BreathingDepthTypes';
import {BreathingDifficultyTypes as BreathingDifficultyTypesEntity} from '../infrastructure/entity/BreathingDifficultyTypes';
import {BreathingDurationTypes as BreathingDurationTypesEntity} from '../infrastructure/entity/BreathingDurationTypes';
import {BreathingTypes, BreathingDepth, BreathingDifficulty, BreathingDuration} from '../domain/BreathingCharacteristics';
import {DatasetAudioTypes as DatasetAudioTypesEntity} from '../infrastructure/entity/DatasetAudioTypes';
import {DatasetAudioTypes} from '../domain/DatasetAudio';
import {DatasetMarkingStatus as DatasetMarkingStatusEntity} from '../infrastructure/entity/DatasetMarkingStatus';
import {DatasetMarkingStatus} from '../domain/DatasetMarkingStatus';
import {GenderTypes} from '../infrastructure/entity/GenderTypes';
import {Gender} from '../domain/Gender';
import {DatasetRequestStatus as EntityDatasetRequestStatus} from '../infrastructure/entity/DatasetRequestStatus';
import {HWRequestStatus as HWRequestStatusEntity} from '../infrastructure/entity/HWRequestStatus';
import {Covid19SymptomaticTypes as EntityCovid19SymptomaticTypes} from '../infrastructure/entity/Covid19SymptomaticTypes';
import {Covid19SymptomaticTypes as DomainCovid19SymptomaticTypes} from '../domain/Covid19Types';
import {DatasetEpisodesTypes as DomainDatasetEpisodesTypes} from '../domain/DatasetEpisodesTypes';
import {DatasetEpisodesTypes as EntityDatasetEpisodesTypes} from '../infrastructure/entity/DatasetEpisodesTypes';
import {TelegramDiagnosticRequestStatus as EntityTgStatus} from '../infrastructure/entity/TelegramDiagnosticRequestStatus';
import {TelegramDiagnosticRequestStatus as DomainTgStatus} from '../domain/RequestStatus';
import {TelegramDatasetRequestStatus as EntityTgDataStatus} from '../infrastructure/entity/TelegramDatasetRequestStatus';
import {TelegramDatasetRequestStatus as DomainTgDataStatus} from '../domain/RequestStatus';
import {TgNewDiagnosticRequestStatus as EntityTgDiagnosticStatus} from '../infrastructure/entity/TgNewDiagnosticRequestStatus';
import {TgNewDiagnosticRequestStatus as DomainTgDiagnosticStatus} from '../domain/RequestStatus';
import {Bots as EntityBots} from '../infrastructure/entity/Bots';
import {Bots as DomainBots} from '../domain/Bots';
import {PaymentTypes as EntityPayments} from '../infrastructure/entity/PaymentTypes';
import {Payments as DomainPayments} from '../domain/Payments';
import {NoiseTypes as EntityNoiseTypes} from '../infrastructure/entity/NoiseTypes';
import {NoiseTypes as DomainNoiseTypes} from '../domain/NoiseTypes';

export const coswara = new User();
export const user = new User();
// Need initialized logins for filters in some db queries
coswara.login = 'coswara';
user.login = 'user';

export type EntityType<T> = ObjectType<T> | EntitySchema<T> | string

export const saveWithConflict = async <U, T extends EntityType<U>>(entity: T, values: QueryDeepPartialEntity<U>, connection: Connection) => {
    return connection
        .createQueryBuilder()
        .insert()
        .into(entity)
        .values(values)
        .onConflict('DO NOTHING')
        .execute();
};

export const seedDb = async (connection: Connection) => {
    const queryRunner = connection.createQueryRunner();
    await queryRunner.startTransaction('REPEATABLE READ');

    const patientRole = new Roles();
    patientRole.role = UserRoleTypes.PATIENT;
    const datasetRole = new Roles();
    datasetRole.role = UserRoleTypes.DATASET;
    const edifierRole = new Roles();
    edifierRole.role = UserRoleTypes.EDIFIER;
    const doctorRole = new Roles();
    doctorRole.role = UserRoleTypes.DOCTOR;
    const dataScientistRole = new Roles();
    dataScientistRole.role = UserRoleTypes.DATA_SCIENTIST;
    const adminRole = new Roles();
    adminRole.role = UserRoleTypes.ADMIN;
    const serverRole = new Roles();
    serverRole.role = UserRoleTypes.EXTERNAL_SERVER;
    const viewerRole = new Roles();
    viewerRole.role = UserRoleTypes.VIEWER;
    await saveWithConflict(Roles, [patientRole, datasetRole, edifierRole, doctorRole, dataScientistRole, adminRole, serverRole, viewerRole], connection);
        
    const diseaseTypes = Object
        .values(DomainDiseaseTypes)
        .map((val) => {
            const entity = new EntityDiseaseTypes();
            entity.disease_type = val;
            return entity;
        });

    const coughIntensityTypes = Object
        .values(DomainCoughIntensityTypes)
        .map((val) => {
            const entity = new EntityCoughIntensityTypes();
            entity.intensity_type = val;
            return entity;
        });

    await connection.createQueryBuilder()
        .insert()
        .into(EntityCoughIntensityTypes)
        .values(coughIntensityTypes)
        .onConflict('DO NOTHING')
    // .onConflict('("id") DO UPDATE SET "intensity_type"=excluded."intensity_type"')
        .execute();

    const acuteTypes = Object
        .values(DomainAcuteCoughTypes)
        .map((val) => {
            const entity = new EntityAcuteCoughTypes();
            entity.acute_cough_types = val;
            return entity;
        });

    const chronicTypes = Object
        .values(DomainChronicCoughTypes)
        .map((val) => {
            const entity = new EntityChornicCoughTypes();
            entity.chronic_cough_type = val;
            return entity;
        });

    await connection.createQueryBuilder()
        .insert()
        .into(EntityDiseaseTypes)
        .values(diseaseTypes)
        .onConflict('DO NOTHING')
        .execute();

    await connection.createQueryBuilder()
        .insert()
        .into(EntityAcuteCoughTypes)
        .values(acuteTypes)
        .onConflict('DO NOTHING')
        .execute();

    await connection.createQueryBuilder()
        .insert()
        .into(EntityChornicCoughTypes)
        .values(chronicTypes)
        .onConflict('DO NOTHING')
        .execute();

    const coughProductivityTypes = Object
        .values(DomainCoughProductivityTypes)
        .map((val) => {
            const entity = new EntityCoughProductivityTypes();
            entity.productivity_type = val;
            return entity;
        });

    await connection.createQueryBuilder()
        .insert()
        .into(EntityCoughProductivityTypes)
        .values(coughProductivityTypes)
        .onConflict('DO NOTHING')
        .execute();

    const diagnosisTypes = Object
        .values(DomainDiagnosisTypes)
        .map((val) => {
            const entity = new EntityDiagnosisTypes();
            entity.diagnosis_type = val;
            return entity;
        });

    await connection.createQueryBuilder()
        .insert()
        .into(EntityDiagnosisTypes)
        .values(diagnosisTypes)
        .onConflict('DO NOTHING')
        .execute();

    const diagnosticRequestStatus = Object
        .values(DomainDiagnosticRequestStatus)
        .map((val) => {
            const entity = new EntityDiagnosticRequestStatus();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(EntityDiagnosticRequestStatus, diagnosticRequestStatus, connection);

    const breathingTypes = Object
        .values(BreathingTypes)
        .map((val) => {
            const entity = new BreathingTypesEntity();
            entity.breathing_type = val;
            return entity;
        });

    await saveWithConflict(BreathingTypesEntity, breathingTypes, connection);

    const breathingDepthTypes = Object
        .values(BreathingDepth)
        .map((val) => {
            const entity = new BreathingDepthTypesEntity();
            entity.depth_type = val;
            return entity;
        });

    await saveWithConflict(BreathingDepthTypesEntity, breathingDepthTypes, connection);

    const breathingDifficultyTypes = Object
        .values(BreathingDifficulty)
        .map((val) => {
            const entity = new BreathingDifficultyTypesEntity();
            entity.difficulty_type = val;
            return entity;
        });

    await saveWithConflict(BreathingDifficultyTypesEntity, breathingDifficultyTypes, connection);

    const breathingDurationTypes = Object
        .values(BreathingDuration)
        .map((val) => {
            const entity = new BreathingDurationTypesEntity();
            entity.duration_type = val;
            return entity;
        });

    await saveWithConflict(BreathingDurationTypesEntity, breathingDurationTypes, connection);

    const genderTypes = Object
        .values(Gender)
        .map((val) => {
            const entity = new GenderTypes();
            entity.gender_type = val;
            return entity;
        });

    await saveWithConflict(GenderTypes, genderTypes, connection);

    const markingStatus = Object
        .values(DatasetMarkingStatus)
        .map((val) => {
            const entity = new DatasetMarkingStatusEntity();
            entity.marking_status = val;
            return entity;
        });

    await saveWithConflict(DatasetMarkingStatusEntity, markingStatus, connection);

    const audioTypes = Object
        .values(DatasetAudioTypes)
        .map((val) => {
            const entity = new DatasetAudioTypesEntity();
            entity.audio_type = val;
            return entity;
        });

    await saveWithConflict(DatasetAudioTypesEntity, audioTypes, connection);

    await connection.createQueryBuilder()
        .insert()
        .into(EntityDiagnosticRequestStatus)
        .values(diagnosticRequestStatus)
        .onConflict('DO NOTHING')
        .execute();

    const hwRequestStatus = Object
        .values(DomainHWRequestStatus)
        .map((val) => {
            const entity = new HWRequestStatusEntity();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(HWRequestStatusEntity, hwRequestStatus, connection);

    const datasetRequestStatus = Object
        .values(DomainDatasetRequestStatus)
        .map((val) => {
            const entity = new EntityDatasetRequestStatus();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(EntityDatasetRequestStatus, datasetRequestStatus, connection);

    const covid19SymptomaticTypes = Object
        .values(DomainCovid19SymptomaticTypes)
        .map((val) => {
            const entity = new EntityCovid19SymptomaticTypes();
            entity.symptomatic_type = val;
            return entity;
        });

    await saveWithConflict(EntityCovid19SymptomaticTypes, covid19SymptomaticTypes, connection);
    
    const datasetEpisodesTypes = Object // Also inserted in migration
        .values(DomainDatasetEpisodesTypes)
        .map(val => {
            const entity = new EntityDatasetEpisodesTypes();
            entity.episode_type = val;
            return entity;
        });
    
    await saveWithConflict(EntityDatasetEpisodesTypes, datasetEpisodesTypes, connection);

    const tgDiagnosticStatus = Object
        .values(DomainTgStatus)
        .map(val => {
            const entity = new EntityTgStatus();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(EntityTgStatus, tgDiagnosticStatus, connection);

    const tgNewDiagnosticStatus = Object
        .values(DomainTgDiagnosticStatus)
        .map(val => {
            const entity = new EntityTgDiagnosticStatus();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(EntityTgDiagnosticStatus, tgNewDiagnosticStatus, connection);

    const tgDatasetStatus = Object
        .values(DomainTgDataStatus)
        .map(val => {
            const entity = new EntityTgDataStatus();
            entity.request_status = val;
            return entity;
        });

    await saveWithConflict(EntityTgDataStatus, tgDatasetStatus, connection);

    const bots = Object
        .values(DomainBots)
        .map(val => {
            const entity = new EntityBots();
            entity.bot_name = val;
            return entity;
        });

    await saveWithConflict(EntityBots, bots, connection);

    const paymentTypes = Object
        .values(DomainPayments)
        .map(val => {
            const entity = new EntityPayments();
            entity.payment_type = val;
            return entity;
        });

    await saveWithConflict(EntityPayments, paymentTypes, connection);

    const noiseTypes = Object
        .values(DomainNoiseTypes)
        .map(val => {
            const entity = new EntityNoiseTypes();
            entity.noise_type = val;
            return entity;
        });

    await saveWithConflict(EntityNoiseTypes, noiseTypes, connection);

    await queryRunner.commitTransaction();

    return Promise.resolve();
};
