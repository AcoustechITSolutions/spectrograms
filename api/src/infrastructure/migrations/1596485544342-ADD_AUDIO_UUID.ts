import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDAUDIOUUID1596485544342 implements MigrationInterface {
    name = 'ADDAUDIOUUID1596485544342'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_audio" ADD "uuid" character varying NOT NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_audio" DROP COLUMN "uuid"');
    }
}
