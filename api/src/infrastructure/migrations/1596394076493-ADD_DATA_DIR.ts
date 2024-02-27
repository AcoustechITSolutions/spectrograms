import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDDATADIR1596394076493 implements MigrationInterface {
    name = 'ADDDATADIR1596394076493'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "data_dir" character varying NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "report_path" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "commentary" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_confirmed" SET DEFAULT false');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_confirmed" DROP DEFAULT');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "commentary" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "report_path" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "data_dir"');
    }
}
