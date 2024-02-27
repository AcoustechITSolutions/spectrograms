import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDEPISODES1596396099804 implements MigrationInterface {
    name = 'ADDEPISODES1596396099804'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."diagnosis_report_episodes" ("id" SERIAL NOT NULL, "episodes_count" integer NOT NULL, "duration_each" double precision array NOT NULL, "mean_duration" double precision NOT NULL, "max_duration" double precision NOT NULL, "min_duration" double precision NOT NULL, "overall_duration" double precision NOT NULL, "reportId" integer, CONSTRAINT "REL_85ec70aab6996d342cb47a7347" UNIQUE ("reportId"), CONSTRAINT "PK_2fef85f3df2d2acd70b612b1c4f" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_85ec70aab6996d342cb47a7347" ON "public"."diagnosis_report_episodes" ("reportId") ');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "diagnosis" character varying(255)');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "diagnosis_probability" double precision');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD "recommendation" character varying');
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" ADD CONSTRAINT "FK_85ec70aab6996d342cb47a73470" FOREIGN KEY ("reportId") REFERENCES "public"."diagnostic_report"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnosis_report_episodes" DROP CONSTRAINT "FK_85ec70aab6996d342cb47a73470"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "recommendation"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "diagnosis_probability"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP COLUMN "diagnosis"');
        await queryRunner.query('DROP INDEX "public"."IDX_85ec70aab6996d342cb47a7347"');
        await queryRunner.query('DROP TABLE "public"."diagnosis_report_episodes"');
    }
}
