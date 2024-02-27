import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDPRIVACYEULAVERSION1599733353924 implements MigrationInterface {
    name = 'ADDPRIVACYEULAVERSION1599733353924'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "privacy_eula_version" integer');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_request_status" ADD CONSTRAINT "UQ_843259f80686302237098ff3d4d" UNIQUE ("request_status")');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "sick_days" DROP NOT NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "sick_days" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_request_status" DROP CONSTRAINT "UQ_843259f80686302237098ff3d4d"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "privacy_eula_version"');
    }
}
