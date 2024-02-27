import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDMARKEDSTATUS1607890110276 implements MigrationInterface {
    name = 'ADDMARKEDSTATUS1607890110276'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "is_marked" boolean NOT NULL DEFAULT false');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "is_marked"');
    }

}
