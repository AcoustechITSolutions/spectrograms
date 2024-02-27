import {MigrationInterface, QueryRunner} from 'typeorm';

export class SCIENTISTMARKING1611938475037 implements MigrationInterface {
    name = 'SCIENTISTMARKING1611938475037'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "is_representative_scientist" boolean');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "is_marked_scientist" boolean NOT NULL DEFAULT false');
        await queryRunner.query('ALTER TABLE "public"."bot_users" ALTER COLUMN "report_language" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."bot_users" ALTER COLUMN "report_language" SET DEFAULT \'ru\'');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."bot_users" ALTER COLUMN "report_language" DROP DEFAULT');
        await queryRunner.query('ALTER TABLE "public"."bot_users" ALTER COLUMN "report_language" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "is_marked_scientist"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "is_representative_scientist"');
    }

}
