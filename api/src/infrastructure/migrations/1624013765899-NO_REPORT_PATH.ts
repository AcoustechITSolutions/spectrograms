import {MigrationInterface, QueryRunner} from "typeorm";

export class NOREPORTPATH1624013765899 implements MigrationInterface {
    name = 'NOREPORTPATH1624013765899'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" DROP COLUMN "report_path"`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" DROP COLUMN "data_dir"`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" DROP COLUMN "recommendation"`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ADD "recommendation" character varying`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ADD "data_dir" character varying NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ADD "report_path" character varying(255)`);
    }

}
