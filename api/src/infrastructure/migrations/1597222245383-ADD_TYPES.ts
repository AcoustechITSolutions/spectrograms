import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDTYPES1597222245383 implements MigrationInterface {
    name = 'ADDTYPES1597222245383'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."breath_audio" ("id" SERIAL NOT NULL, "file_path" character varying NOT NULL, "request_id" integer NOT NULL, CONSTRAINT "REL_c50a0c7f68aff20e8820e16896" UNIQUE ("request_id"), CONSTRAINT "PK_517ede88b3e2962dddaa2bc6b5c" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "acute_cough_types_id" integer');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "chronic_cough_types_id" integer');
        await queryRunner.query('ALTER TABLE "public"."breath_audio" ADD CONSTRAINT "FK_c50a0c7f68aff20e8820e168963" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_b80985f2b28eb64cc33da447ad4" FOREIGN KEY ("acute_cough_types_id") REFERENCES "public"."acute_cough_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_ee58734eb129a019ada294d453b" FOREIGN KEY ("chronic_cough_types_id") REFERENCES "public"."chronic_cough_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'intensive\' TO \'paroxysmal\' ');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'not_intensive\' TO \'not_paroxysmal\' ');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'weak\' TO \'paroxysmal_hacking\' ');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_ee58734eb129a019ada294d453b"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_b80985f2b28eb64cc33da447ad4"');
        await queryRunner.query('ALTER TABLE "public"."breath_audio" DROP CONSTRAINT "FK_c50a0c7f68aff20e8820e168963"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "chronic_cough_types_id"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "acute_cough_types_id"');
        await queryRunner.query('DROP TABLE "public"."breath_audio"');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'paroxysmal\' TO \'intensive\' ');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'not_paroxysmal\' TO \'not_intensive\' ');
        await queryRunner.query('ALTER TYPE "public"."cough_intensity_types_intensity_type_enum" RENAME VALUE \'paroxysmal_hacking\' TO \'weak\'');
    }
}
