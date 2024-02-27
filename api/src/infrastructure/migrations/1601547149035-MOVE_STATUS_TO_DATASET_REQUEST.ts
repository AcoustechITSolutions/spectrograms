import {MigrationInterface, QueryRunner} from 'typeorm';

export class MOVESTATUSTODATASETREQUEST1601547149035 implements MigrationInterface {
    name = 'MOVESTATUSTODATASETREQUEST1601547149035'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_50201d631c0f1c9881421fdb91c"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP CONSTRAINT "FK_b9a9b3c20abcfe9fa72685b9f4c"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP CONSTRAINT "FK_471c6461d1b75ebb7dff96d27a5"');
        await queryRunner.query('DROP INDEX "public"."IDX_471c6461d1b75ebb7dff96d27a"');
        await queryRunner.query('DROP INDEX "public"."IDX_b9a9b3c20abcfe9fa72685b9f4"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "marking_status_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "doctor_status_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD "marking_status_id" integer NOT NULL DEFAULT 2');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD "doctor_status_id" integer NOT NULL DEFAULT 2');
        await queryRunner.query('CREATE INDEX "IDX_fc51b6792323b54950393b076c" ON "public"."dataset_request" ("marking_status_id") ');
        await queryRunner.query('CREATE INDEX "IDX_e8fa9a5ea426ac5cde060bde25" ON "public"."dataset_request" ("doctor_status_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_50201d631c0f1c9881421fdb91c" FOREIGN KEY ("status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_fc51b6792323b54950393b076c5" FOREIGN KEY ("marking_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254" FOREIGN KEY ("doctor_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_fc51b6792323b54950393b076c5"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_50201d631c0f1c9881421fdb91c"');
        await queryRunner.query('DROP INDEX "public"."IDX_e8fa9a5ea426ac5cde060bde25"');
        await queryRunner.query('DROP INDEX "public"."IDX_fc51b6792323b54950393b076c"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP COLUMN "doctor_status_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP COLUMN "marking_status_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "doctor_status_id" integer NOT NULL DEFAULT 2');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "marking_status_id" integer NOT NULL DEFAULT 2');
        await queryRunner.query('CREATE INDEX "IDX_b9a9b3c20abcfe9fa72685b9f4" ON "public"."dataset_audio_info" ("doctor_status_id") ');
        await queryRunner.query('CREATE INDEX "IDX_471c6461d1b75ebb7dff96d27a" ON "public"."dataset_audio_info" ("marking_status_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD CONSTRAINT "FK_471c6461d1b75ebb7dff96d27a5" FOREIGN KEY ("marking_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD CONSTRAINT "FK_b9a9b3c20abcfe9fa72685b9f4c" FOREIGN KEY ("doctor_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_50201d631c0f1c9881421fdb91c" FOREIGN KEY ("status_id") REFERENCES "public"."dataset_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
