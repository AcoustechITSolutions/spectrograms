import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDDATASETSTATUS1600869783504 implements MigrationInterface {
    name = 'ADDDATASETSTATUS1600869783504'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."dataset_request_status_request_status_enum" AS ENUM(\'preprocessing\', \'preprocessing_error\', \'pending\', \'error\', \'done\', \'creating_error\')');
        await queryRunner.query('CREATE TABLE "public"."dataset_request_status" ("id" SERIAL NOT NULL, "request_status" "public"."dataset_request_status_request_status_enum" NOT NULL, CONSTRAINT "UQ_6b133eaf56df3abe4d67b1c5b9b" UNIQUE ("request_status"), CONSTRAINT "PK_50201d631c0f1c9881421fdb91c" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD "status_id" integer NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_cough_characteristics" ALTER COLUMN "symptom_duration" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_cough_characteristics" ALTER COLUMN "commentary" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "samplerate" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "spectrogram_path" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "audio_duration" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "audio_duration" TYPE double precision');
        await queryRunner.query('CREATE INDEX "IDX_50201d631c0f1c9881421fdb91" ON "public"."dataset_request" ("status_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_50201d631c0f1c9881421fdb91c" FOREIGN KEY ("status_id") REFERENCES "public"."dataset_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_50201d631c0f1c9881421fdb91c"');
        await queryRunner.query('DROP INDEX "public"."IDX_50201d631c0f1c9881421fdb91"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "audio_duration" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "audio_duration" TYPE integer');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "spectrogram_path" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ALTER COLUMN "samplerate" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_cough_characteristics" ALTER COLUMN "commentary" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_cough_characteristics" ALTER COLUMN "symptom_duration" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP COLUMN "status_id"');
        await queryRunner.query('DROP TABLE "public"."dataset_request_status"');
        await queryRunner.query('DROP TYPE "public"."dataset_request_status_request_status_enum"');
    }
}
