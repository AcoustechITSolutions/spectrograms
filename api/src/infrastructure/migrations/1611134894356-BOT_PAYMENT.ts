import {MigrationInterface, QueryRunner} from 'typeorm';

export class BOTPAYMENT1611134894356 implements MigrationInterface {
    name = 'BOTPAYMENT1611134894356'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."bot_users" ("id" SERIAL NOT NULL, "chat_id" integer NOT NULL, "report_language" character varying, CONSTRAINT "UQ_747facce64d2ad73343fa25ade0" UNIQUE ("chat_id"), CONSTRAINT "PK_b6966bd7c371731b005efb0fdad" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_747facce64d2ad73343fa25ade" ON "public"."bot_users" ("chat_id") ');
        await queryRunner.query('INSERT INTO "bot_users"("chat_id", "report_language") SELECT DISTINCT "chat_id", "report_language" FROM "tg_dataset_request" ORDER BY "report_language" ON CONFLICT DO NOTHING');
        await queryRunner.query('INSERT INTO "bot_users"("chat_id", "report_language") SELECT DISTINCT "chat_id", "report_language" FROM "tg_diagnostic_request" ORDER BY "report_language" ON CONFLICT DO NOTHING'); 
        await queryRunner.query('CREATE TYPE "public"."bots_bot_name_enum" AS ENUM(\'tg_cough_analysis\')');
        await queryRunner.query('CREATE TABLE "public"."bots" ("id" SERIAL NOT NULL, "bot_name" "public"."bots_bot_name_enum" NOT NULL, CONSTRAINT "UQ_a80dcd55fd19728486152e52da8" UNIQUE ("bot_name"), CONSTRAINT "PK_39c09e8be25f3b90b8f74b327d8" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."payment_types_payment_type_enum" AS ENUM(\'single\', \'unlimited\')');
        await queryRunner.query('CREATE TABLE "public"."payment_types" ("id" SERIAL NOT NULL, "payment_type" "public"."payment_types_payment_type_enum" NOT NULL, CONSTRAINT "UQ_6ae3313dcdd6e61f715468bf689" UNIQUE ("payment_type"), CONSTRAINT "PK_fe6a5a639fdb5f6625f25a60b97" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."bot_payments" ("id" SERIAL NOT NULL, "chat_id" integer NOT NULL, "bot_id" integer NOT NULL, "payment_type_id" integer NOT NULL, "payment_sum" integer, "is_active" boolean NOT NULL DEFAULT \'false\', CONSTRAINT "PK_661b51578cf9d35502a106cb311" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_7329447013d87852d848fbea11" ON "public"."bot_payments" ("chat_id") ');
        await queryRunner.query('CREATE INDEX "IDX_091abaf937b6c9e73c6a3c4403" ON "public"."bot_payments" ("bot_id") ');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" DROP COLUMN "report_language"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP COLUMN "report_language"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP COLUMN "payment_sum"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP COLUMN "paid"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP COLUMN "report_language"');
        await queryRunner.query('ALTER TYPE "public"."tg_new_diagnostic_request_status_request_status_enum" RENAME TO "tg_new_diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."tg_new_diagnostic_request_status_request_status_enum" AS ENUM(\'start\', \'support\', \'payment\', \'disclaimer\', \'conditions\', \'age\', \'gender\', \'is_smoking\', \'cough_audio\', \'done\', \'cancelled\', \'language\')');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request_status" ALTER COLUMN "request_status" TYPE "public"."tg_new_diagnostic_request_status_request_status_enum" USING "request_status"::"text"::"public"."tg_new_diagnostic_request_status_request_status_enum"');
        await queryRunner.query('DROP TYPE "public"."tg_new_diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('CREATE INDEX "IDX_fa0d32790f0a76371a0cb938a0" ON "public"."tg_dataset_request" ("chat_id") ');
        await queryRunner.query('CREATE INDEX "IDX_fc3ba1c32261c3deee89051d61" ON "public"."tg_diagnostic_request" ("chat_id") ');
        await queryRunner.query('CREATE INDEX "IDX_631999ce74a16a07ca21ab503f" ON "public"."tg_new_diagnostic_request" ("chat_id") ');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ADD CONSTRAINT "FK_7329447013d87852d848fbea11e" FOREIGN KEY ("chat_id") REFERENCES "public"."bot_users"("chat_id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ADD CONSTRAINT "FK_091abaf937b6c9e73c6a3c4403b" FOREIGN KEY ("bot_id") REFERENCES "public"."bots"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ADD CONSTRAINT "FK_9cbc5409685adc6cbfb7069b973" FOREIGN KEY ("payment_type_id") REFERENCES "public"."payment_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" ADD CONSTRAINT "FK_fa0d32790f0a76371a0cb938a0f" FOREIGN KEY ("chat_id") REFERENCES "public"."bot_users"("chat_id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD CONSTRAINT "FK_fc3ba1c32261c3deee89051d61b" FOREIGN KEY ("chat_id") REFERENCES "public"."bot_users"("chat_id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD CONSTRAINT "FK_631999ce74a16a07ca21ab503f6" FOREIGN KEY ("chat_id") REFERENCES "public"."bot_users"("chat_id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP CONSTRAINT "FK_631999ce74a16a07ca21ab503f6"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP CONSTRAINT "FK_fc3ba1c32261c3deee89051d61b"');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" DROP CONSTRAINT "FK_fa0d32790f0a76371a0cb938a0f"');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" DROP CONSTRAINT "FK_9cbc5409685adc6cbfb7069b973"');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" DROP CONSTRAINT "FK_091abaf937b6c9e73c6a3c4403b"');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" DROP CONSTRAINT "FK_7329447013d87852d848fbea11e"');
        await queryRunner.query('DROP INDEX "public"."IDX_631999ce74a16a07ca21ab503f"');
        await queryRunner.query('DROP INDEX "public"."IDX_fc3ba1c32261c3deee89051d61"');
        await queryRunner.query('DROP INDEX "public"."IDX_fa0d32790f0a76371a0cb938a0"');
        await queryRunner.query('CREATE TYPE "public"."tg_new_diagnostic_request_status_request_status_enum_old" AS ENUM(\'start\', \'support\', \'payment\', \'disclaimer\', \'conditions\', \'age\', \'gender\', \'is_smoking\', \'cough_audio\', \'done\', \'cancelled\')');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request_status" ALTER COLUMN "request_status" TYPE "public"."tg_new_diagnostic_request_status_request_status_enum_old" USING "request_status"::"text"::"public"."tg_new_diagnostic_request_status_request_status_enum_old"');
        await queryRunner.query('DROP TYPE "public"."tg_new_diagnostic_request_status_request_status_enum"');
        await queryRunner.query('ALTER TYPE "public"."tg_new_diagnostic_request_status_request_status_enum_old" RENAME TO  "tg_new_diagnostic_request_status_request_status_enum"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD "report_language" character varying');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD "paid" boolean NOT NULL DEFAULT false');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD "payment_sum" integer');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD "report_language" character varying');
        await queryRunner.query('ALTER TABLE "public"."tg_dataset_request" ADD "report_language" character varying');
        await queryRunner.query('DROP INDEX "public"."IDX_091abaf937b6c9e73c6a3c4403"');
        await queryRunner.query('DROP INDEX "public"."IDX_7329447013d87852d848fbea11"');
        await queryRunner.query('DROP TABLE "public"."bot_payments"');
        await queryRunner.query('DROP TABLE "public"."payment_types"');
        await queryRunner.query('DROP TYPE "public"."payment_types_payment_type_enum"');
        await queryRunner.query('DROP TABLE "public"."bots"');
        await queryRunner.query('DROP TYPE "public"."bots_bot_name_enum"');
        await queryRunner.query('DROP INDEX "public"."IDX_747facce64d2ad73343fa25ade"');
        await queryRunner.query('DROP TABLE "public"."bot_users"');
    }

}
