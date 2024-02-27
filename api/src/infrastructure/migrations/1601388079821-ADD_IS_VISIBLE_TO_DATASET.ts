import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDISVISIBLETODATASET1601388079821 implements MigrationInterface {
    name = 'ADDISVISIBLETODATASET1601388079821'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD "is_visible" boolean NOT NULL DEFAULT true');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP COLUMN "is_visible"');
    }
}
