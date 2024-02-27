import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDTELEGRAMBOTREQUEST1604002481920 implements MigrationInterface {
    name = 'ADDTELEGRAMBOTREQUEST1604002481920'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."tg_diagnostic_requests_status_request_status_enum" AS ENUM(\'age\', \'gender\', \'is_smoking\', \'is_forced\', \'cough_audio\', \'done\', \'cancelled\')');
        await queryRunner.query('CREATE TABLE "public"."tg_diagnostic_requests_status" ("id" SERIAL NOT NULL, "request_status" "public"."tg_diagnostic_requests_status_request_status_enum" NOT NULL, CONSTRAINT "UQ_b07d1226a0fd7c9462097d573f6" UNIQUE ("request_status"), CONSTRAINT "PK_9c48369f5915a89736741a928eb" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."tg_diagnostic_request" ("id" SERIAL NOT NULL, "chat_id" integer NOT NULL, "status_id" integer NOT NULL, "dateCreated" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, "age" integer, "gender_id" integer, "is_smoking" boolean, "cough_audio_path" character varying, "is_forced" boolean, CONSTRAINT "PK_92d48201bacd1a2554a30f780a8" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD CONSTRAINT "FK_4b48faf7fba2a42807391d18726" FOREIGN KEY ("status_id") REFERENCES "public"."tg_diagnostic_requests_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD CONSTRAINT "FK_1491599c46315b1334966801c55" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP CONSTRAINT "FK_1491599c46315b1334966801c55"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP CONSTRAINT "FK_4b48faf7fba2a42807391d18726"');
        await queryRunner.query('DROP TABLE "public"."tg_diagnostic_request"');
        await queryRunner.query('DROP TABLE "public"."tg_diagnostic_requests_status"');
        await queryRunner.query('DROP TYPE "public"."tg_diagnostic_requests_status_request_status_enum"');
    }

}
