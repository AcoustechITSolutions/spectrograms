import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDISREPRESENTATIVE1605012604578 implements MigrationInterface {
    name = 'ADDISREPRESENTATIVE1605012604578'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "is_validation_audio" boolean NOT NULL DEFAULT false');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "is_validation_audio"');
    }

}
