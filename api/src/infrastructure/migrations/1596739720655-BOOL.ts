import {MigrationInterface, QueryRunner} from 'typeorm';

export class BOOL1596739720655 implements MigrationInterface {
    name = 'BOOL1596739720655'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "is_user_visible"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "is_visible" boolean NOT NULL DEFAULT true');

        // await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "is_user_visible" TO "is_visible"`);
        // await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_visible" SET DEFAULT true`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "is_visible"');

        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "is_user_visible" boolean NOT NULL');
        // await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_visible" DROP DEFAULT`);
        // await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "is_visible" TO "is_user_visible"`);
    }
}
