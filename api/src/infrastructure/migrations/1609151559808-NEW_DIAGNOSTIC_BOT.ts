import {MigrationInterface, QueryRunner} from 'typeorm';

export class NEWDIAGNOSTICBOT1609151559808 implements MigrationInterface {
    name = 'NEWDIAGNOSTICBOT1609151559808'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."tg_new_diagnostic_request_status_request_status_enum" AS ENUM(\'start\', \'support\', \'payment\', \'disclaimer\', \'conditions\', \'age\', \'gender\', \'is_smoking\', \'cough_audio\', \'done\', \'cancelled\')');
        await queryRunner.query('CREATE TABLE "public"."tg_new_diagnostic_request_status" ("id" SERIAL NOT NULL, "request_status" "public"."tg_new_diagnostic_request_status_request_status_enum" NOT NULL, CONSTRAINT "UQ_8f031f688239f24e639a4a7b04c" UNIQUE ("request_status"), CONSTRAINT "PK_55daf2a323a084b986f4044a4a9" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."tg_new_diagnostic_request" ("id" SERIAL NOT NULL, "request_id" integer, "chat_id" integer NOT NULL, "status_id" integer NOT NULL, "dateCreated" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, "payment_sum" integer, "paid" boolean NOT NULL DEFAULT \'false\', "age" integer, "gender_id" integer, "is_smoking" boolean, "cough_audio_path" character varying, "report_language" character varying, CONSTRAINT "REL_bd636ba0b38819feecaea2b904" UNIQUE ("request_id"), CONSTRAINT "PK_cbdbdb382a7540b6a896ff94f3d" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_bd636ba0b38819feecaea2b904" ON "public"."tg_new_diagnostic_request" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD CONSTRAINT "FK_bd636ba0b38819feecaea2b9048" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD CONSTRAINT "FK_55daf2a323a084b986f4044a4a9" FOREIGN KEY ("status_id") REFERENCES "public"."tg_new_diagnostic_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD CONSTRAINT "FK_7f4baa1d439ce6d3d47eb03bcca" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP CONSTRAINT "FK_7f4baa1d439ce6d3d47eb03bcca"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP CONSTRAINT "FK_55daf2a323a084b986f4044a4a9"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP CONSTRAINT "FK_bd636ba0b38819feecaea2b9048"');
        await queryRunner.query('DROP INDEX "public"."IDX_bd636ba0b38819feecaea2b904"');
        await queryRunner.query('DROP TABLE "public"."tg_new_diagnostic_request"');
        await queryRunner.query('DROP TABLE "public"."tg_new_diagnostic_request_status"');
        await queryRunner.query('DROP TYPE "public"."tg_new_diagnostic_request_status_request_status_enum"');
    }

}
