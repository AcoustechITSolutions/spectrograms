import {MigrationInterface, QueryRunner} from 'typeorm';

export class USERCOMMENT1608049636313 implements MigrationInterface {
    name = 'USERCOMMENT1608049636313'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."users" ADD "comment" character varying');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_dataset_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_dataset_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('ALTER TABLE "public"."users" DROP COLUMN "comment"');
    }

}
