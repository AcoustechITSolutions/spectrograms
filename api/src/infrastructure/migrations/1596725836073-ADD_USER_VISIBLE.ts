import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDUSERVISIBLE1596725836073 implements MigrationInterface {
    name = 'ADDUSERVISIBLE1596725836073'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "is_user_visible" boolean NOT NULL DEFAULT true');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "is_user_visible"');
    }
}
