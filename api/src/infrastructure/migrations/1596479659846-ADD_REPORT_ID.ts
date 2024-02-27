import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDREPORTID1596479659846 implements MigrationInterface {
    name = 'ADDREPORTID1596479659846'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" DROP CONSTRAINT "FK_85ec70aab6996d342cb47a73470"');
        await queryRunner.query('DROP INDEX "public"."IDX_85ec70aab6996d342cb47a7347"');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" RENAME COLUMN "reportId" TO "report_id"');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" RENAME CONSTRAINT "REL_85ec70aab6996d342cb47a7347" TO "UQ_de39fcae7c6dd907317eda33666"');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" ALTER COLUMN "report_id" SET NOT NULL');
        await queryRunner.query('CREATE INDEX "IDX_de39fcae7c6dd907317eda3366" ON "public"."diagnosis_report_episodes" ("report_id") ');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" ADD CONSTRAINT "FK_de39fcae7c6dd907317eda33666" FOREIGN KEY ("report_id") REFERENCES "public"."diagnostic_report"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" DROP CONSTRAINT "FK_de39fcae7c6dd907317eda33666"');
        await queryRunner.query('DROP INDEX "public"."IDX_de39fcae7c6dd907317eda3366"');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" ALTER COLUMN "report_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" RENAME CONSTRAINT "UQ_de39fcae7c6dd907317eda33666" TO "REL_85ec70aab6996d342cb47a7347"');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" RENAME COLUMN "report_id" TO "reportId"');
        await queryRunner.query('CREATE INDEX "IDX_85ec70aab6996d342cb47a7347" ON "public"."diagnosis_report_episodes" ("reportId") ');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" ADD CONSTRAINT "FK_85ec70aab6996d342cb47a73470" FOREIGN KEY ("reportId") REFERENCES "public"."diagnostic_report"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
