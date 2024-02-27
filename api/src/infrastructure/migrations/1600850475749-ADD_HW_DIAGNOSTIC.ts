import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDHWDIAGNOSTIC1600850475749 implements MigrationInterface {
    name = 'ADDHWDIAGNOSTIC1600850475749'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."hw_diagnostic_request_status_request_status_enum" AS ENUM(\'error\', \'processing\', \'success\')');
        await queryRunner.query('CREATE TABLE "public"."hw_diagnostic_request_status" ("id" SERIAL NOT NULL, "request_status" "public"."hw_diagnostic_request_status_request_status_enum" NOT NULL, CONSTRAINT "UQ_1da65db84a33f2303bdcf4f1d3e" UNIQUE ("request_status"), CONSTRAINT "PK_78e2b1ef70b41c092d8d20c4e9c" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."hw_diagnostic_request" ("id" SERIAL NOT NULL, "date_created" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, "user_id" integer NOT NULL, "status_id" integer NOT NULL, CONSTRAINT "PK_c1a859d848faf01966a8e8201b3" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_ce8578c965e6308bd0f388f794" ON "public"."hw_diagnostic_request" ("user_id") ');
        await queryRunner.query('CREATE INDEX "IDX_78e2b1ef70b41c092d8d20c4e9" ON "public"."hw_diagnostic_request" ("status_id") ');
        await queryRunner.query('CREATE TABLE "public"."hw_cough_audio" ("id" SERIAL NOT NULL, "file_path" character varying NOT NULL, "request_id" integer NOT NULL, "samplerate" integer, "duration" double precision, CONSTRAINT "REL_8bc85a48895c9598ab02654150" UNIQUE ("request_id"), CONSTRAINT "PK_f809741f2620ff925520791ed05" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."hw_diagnostic_report" ("id" SERIAL NOT NULL, "request_id" integer NOT NULL, "diagnosis_probability" double precision, CONSTRAINT "REL_8babf5c5552c5c042459a8b1c0" UNIQUE ("request_id"), CONSTRAINT "PK_334b3979b301c67d6d6774bc8e9" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_8babf5c5552c5c042459a8b1c0" ON "public"."hw_diagnostic_report" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_request" ADD CONSTRAINT "FK_ce8578c965e6308bd0f388f794c" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_request" ADD CONSTRAINT "FK_78e2b1ef70b41c092d8d20c4e9c" FOREIGN KEY ("status_id") REFERENCES "public"."hw_diagnostic_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."hw_cough_audio" ADD CONSTRAINT "FK_8bc85a48895c9598ab026541503" FOREIGN KEY ("request_id") REFERENCES "public"."hw_diagnostic_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_report" ADD CONSTRAINT "FK_8babf5c5552c5c042459a8b1c0e" FOREIGN KEY ("request_id") REFERENCES "public"."hw_diagnostic_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_report" DROP CONSTRAINT "FK_8babf5c5552c5c042459a8b1c0e"');
        await queryRunner.query('ALTER TABLE "public"."hw_cough_audio" DROP CONSTRAINT "FK_8bc85a48895c9598ab026541503"');
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_request" DROP CONSTRAINT "FK_78e2b1ef70b41c092d8d20c4e9c"');
        await queryRunner.query('ALTER TABLE "public"."hw_diagnostic_request" DROP CONSTRAINT "FK_ce8578c965e6308bd0f388f794c"');
        await queryRunner.query('DROP INDEX "public"."IDX_8babf5c5552c5c042459a8b1c0"');
        await queryRunner.query('DROP TABLE "public"."hw_diagnostic_report"');
        await queryRunner.query('DROP TABLE "public"."hw_cough_audio"');
        await queryRunner.query('DROP INDEX "public"."IDX_78e2b1ef70b41c092d8d20c4e9"');
        await queryRunner.query('DROP INDEX "public"."IDX_ce8578c965e6308bd0f388f794"');
        await queryRunner.query('DROP TABLE "public"."hw_diagnostic_request"');
        await queryRunner.query('DROP TABLE "public"."hw_diagnostic_request_status"');
        await queryRunner.query('DROP TYPE "public"."hw_diagnostic_request_status_request_status_enum"');
    }
}
