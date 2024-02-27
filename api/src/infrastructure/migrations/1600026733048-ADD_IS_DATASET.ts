import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDISDATASET1600026733048 implements MigrationInterface {
    name = 'ADDISDATASET1600026733048'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" ADD COLUMN "is_dataset" boolean NOT NULL DEFAULT false');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" DROP COLUMN "is_dataset"');
    }
}
