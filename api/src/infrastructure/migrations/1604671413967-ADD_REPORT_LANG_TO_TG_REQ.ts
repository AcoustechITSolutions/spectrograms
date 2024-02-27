import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDREPORTLANGTOTGREQ1604671413967 implements MigrationInterface {
    name = 'ADDREPORTLANGTOTGREQ1604671413967'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD "report_language" character varying');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP COLUMN "report_language"');
    }
    
}
