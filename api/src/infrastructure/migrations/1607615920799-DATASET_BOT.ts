import {MigrationInterface, QueryRunner} from 'typeorm';

export class DATASETBOT1607615920799 implements MigrationInterface {
    name = 'DATASETBOT1607615920799'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."tg_dataset_requests_status_request_status_enum" AS ENUM(\'age\', \'gender\', \'is_smoking\', \'is_covid\', \'is_disease\', \'disease_name\', \'cough_audio\', \'is_forced\', \'done\', \'cancelled\')');
        await queryRunner.query('CREATE TABLE "public"."tg_dataset_requests_status" ("id" SERIAL NOT NULL, "request_status" "public"."tg_dataset_requests_status_request_status_enum" NOT NULL, CONSTRAINT "UQ_18ad510a05fc60488cab9c9a173" UNIQUE ("request_status"), CONSTRAINT "PK_36cb15700c471d44eb087db5fb2" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."tg_dataset_request" ("id" SERIAL NOT NULL, "request_id" integer, "chat_id" integer NOT NULL, "status_id" integer NOT NULL, "dateCreated" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, "age" integer, "gender_id" integer, "is_smoking" boolean, "is_covid" boolean, "is_disease" boolean, "disease_name" character varying, "cough_audio_path" character varying, "is_forced" boolean, "report_language" character varying, CONSTRAINT "REL_1fab7d9a00848a2a1ae5ef7a5b" UNIQUE ("request_id"), CONSTRAINT "PK_5068664a667b59d37ebb2a41629" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_1fab7d9a00848a2a1ae5ef7a5b" ON "public"."tg_dataset_request" ("request_id") ');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" ADD CONSTRAINT "FK_1fab7d9a00848a2a1ae5ef7a5b9" FOREIGN KEY ("request_id") REFERENCES "public"."dataset_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" ADD CONSTRAINT "FK_b03c8adc4bcee59ee7ea6c32d4b" FOREIGN KEY ("status_id") REFERENCES "public"."tg_dataset_requests_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" ADD CONSTRAINT "FK_2b21a4e017830f4cd84fc702d95" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" DROP CONSTRAINT "FK_2b21a4e017830f4cd84fc702d95"');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" DROP CONSTRAINT "FK_b03c8adc4bcee59ee7ea6c32d4b"');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" DROP CONSTRAINT "FK_1fab7d9a00848a2a1ae5ef7a5b9"');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('DROP INDEX "public"."IDX_1fab7d9a00848a2a1ae5ef7a5b"');
        await queryRunner.query('DROP TABLE "public"."tg_dataset_request"');
        await queryRunner.query('DROP TABLE "public"."tg_dataset_requests_status"');
        await queryRunner.query('DROP TYPE "public"."tg_dataset_requests_status_request_status_enum"');
    }

}
