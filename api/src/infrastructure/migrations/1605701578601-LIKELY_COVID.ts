import {MigrationInterface, QueryRunner} from 'typeorm';

export class LIKELYCOVID1605701578601 implements MigrationInterface {
    name = 'LIKELYCOVID1605701578601'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('ALTER TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum" RENAME TO "covid19_symptomatic_types_symptomatic_type_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum" AS ENUM(\'no_covid19\', \'likely_covid19\', \'covid19_without_symptomatic\', \'covid19_mild_symptomatic\', \'covid19_severe_symptomatic\')');
        await queryRunner.query('ALTER TABLE "public"."covid19_symptomatic_types" ALTER COLUMN "symptomatic_type" TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum" USING "symptomatic_type"::"text"::"public"."covid19_symptomatic_types_symptomatic_type_enum"');
        await queryRunner.query('DROP TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum_old"');
        await queryRunner.query('COMMENT ON COLUMN "public"."covid19_symptomatic_types"."symptomatic_type" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."covid19_symptomatic_types"."symptomatic_type" IS NULL');
        await queryRunner.query('CREATE TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum_old" AS ENUM(\'no_covid19\', \'covid19_without_symptomatic\', \'covid19_mild_symptomatic\', \'covid19_severe_symptomatic\')');
        await queryRunner.query('ALTER TABLE "public"."covid19_symptomatic_types" ALTER COLUMN "symptomatic_type" TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum_old" USING "symptomatic_type"::"text"::"public"."covid19_symptomatic_types_symptomatic_type_enum_old"');
        await queryRunner.query('DROP TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum"');
        await queryRunner.query('ALTER TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum_old" RENAME TO  "covid19_symptomatic_types_symptomatic_type_enum"');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
    }

}
