import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXCOUGHTYPES1596701605566 implements MigrationInterface {
    name = 'FIXCOUGHTYPES1596701605566'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "diagnosis" SET DATA TYPE integer USING CASE WHEN  diagnosis = \'Здоров\' THEN 0 WHEN diagnosis = \'В зоне риска\' THEN 1 ELSE 2 END');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "diagnosis" TO "diagnosis_id"');
        await queryRunner.query('CREATE TYPE "public"."diagnosis_types_diagnosis_type_enum" AS ENUM(\'healthy\', \'at_risk\', \'covid_19\')');
        await queryRunner.query('CREATE TABLE "public"."diagnosis_types" ("id" SERIAL NOT NULL, "diagnosis_type" "public"."diagnosis_types_diagnosis_type_enum" NOT NULL, CONSTRAINT "UQ_b06339f79617b29ae150322149a" UNIQUE ("diagnosis_type"), CONSTRAINT "PK_e33133dd01de4cb6cbc59b86fee" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP COLUMN "transitivity_id"');
        await queryRunner.query('ALTER TYPE "public"."cough_productivity_types_productivity_type_enum" RENAME TO "cough_productivity_types_productivity_type_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."cough_productivity_types_productivity_type_enum" AS ENUM(\'productive\', \'unproductive\', \'wet_productive_small\', \'dry_productive_small\')');
        await queryRunner.query('ALTER TABLE "public"."cough_productivity_types" ALTER COLUMN "productivity_type" TYPE "public"."cough_productivity_types_productivity_type_enum" USING "productivity_type"::"text"::"public"."cough_productivity_types_productivity_type_enum"');
        await queryRunner.query('DROP TYPE "public"."cough_productivity_types_productivity_type_enum_old"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD CONSTRAINT "FK_e24ebf91d488e46f44bf8a138b7" FOREIGN KEY ("diagnosis_id") REFERENCES "public"."diagnosis_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP CONSTRAINT "FK_e24ebf91d488e46f44bf8a138b7"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "diagnosis_id" SET DATA TYPE character varying(255) USING CASE WHEN  diagnosis = 0 THEN \'Здоров\' WHEN diagnosis = 1 THEN \'В зоне риска\' ELSE \'COVID-19\' END');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "diagnosis_id" TO "diagnosis"');
        await queryRunner.query('CREATE TYPE "public"."cough_productivity_types_productivity_type_enum_old" AS ENUM(\'productive\', \'unproductive\')');
        await queryRunner.query('ALTER TABLE "public"."cough_productivity_types" ALTER COLUMN "productivity_type" TYPE "public"."cough_productivity_types_productivity_type_enum_old" USING "productivity_type"::"text"::"public"."cough_productivity_types_productivity_type_enum_old"');
        await queryRunner.query('DROP TYPE "public"."cough_productivity_types_productivity_type_enum"');
        await queryRunner.query('ALTER TYPE "public"."cough_productivity_types_productivity_type_enum_old" RENAME TO  "cough_productivity_types_productivity_type_enum"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD "transitivity_id" integer');
        await queryRunner.query('DROP TABLE "public"."diagnosis_types"');
        await queryRunner.query('DROP TYPE "public"."diagnosis_types_diagnosis_type_enum"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb" FOREIGN KEY ("transitivity_id") REFERENCES "public"."cough_transitivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
