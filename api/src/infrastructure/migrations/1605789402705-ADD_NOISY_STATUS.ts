import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDNOISYSTATUS1605789402705 implements MigrationInterface {
    name = 'ADDNOISYSTATUS1605789402705'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TYPE "public"."diagnostic_request_status_request_status_enum" RENAME TO "diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."diagnostic_request_status_request_status_enum" AS ENUM(\'pending\', \'error\', \'processing\', \'success\', \'noisy_audio\')');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_request_status" ALTER COLUMN "request_status" TYPE "public"."diagnostic_request_status_request_status_enum" USING "request_status"::"text"::"public"."diagnostic_request_status_request_status_enum"');
        await queryRunner.query('DROP TYPE "public"."diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_request_status"."request_status" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_request_status"."request_status" IS NULL');
        await queryRunner.query('CREATE TYPE "public"."diagnostic_request_status_request_status_enum_old" AS ENUM(\'pending\', \'error\', \'processing\', \'success\')');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_request_status" ALTER COLUMN "request_status" TYPE "public"."diagnostic_request_status_request_status_enum_old" USING "request_status"::"text"::"public"."diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('DROP TYPE "public"."diagnostic_request_status_request_status_enum"');
        await queryRunner.query('ALTER TYPE "public"."diagnostic_request_status_request_status_enum_old" RENAME TO  "diagnostic_request_status_request_status_enum"');
    }

}
