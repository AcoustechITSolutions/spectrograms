import {MigrationInterface, QueryRunner} from 'typeorm';

export class TRANSACTIONS1612455112210 implements MigrationInterface {
    name = 'TRANSACTIONS1612455112210'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."payonline_transactions" ("id" SERIAL NOT NULL, "transaction_id" integer NOT NULL, "date_time" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "order_id" integer NOT NULL, "bot_id" integer NOT NULL, "amount" integer NOT NULL, "currency" character varying NOT NULL, "security_key" character varying NOT NULL, "is_confirmed" boolean NOT NULL, CONSTRAINT "PK_80d56590a112050db6adc0c7c40" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_40002c953ca3fde74084b8c2f6" ON "public"."payonline_transactions" ("order_id") ');
        await queryRunner.query('CREATE INDEX "IDX_0a99c3b353c97edb0dd3d53726" ON "public"."payonline_transactions" ("bot_id") ');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD CONSTRAINT "FK_40002c953ca3fde74084b8c2f6e" FOREIGN KEY ("order_id") REFERENCES "public"."tg_new_diagnostic_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD CONSTRAINT "FK_0a99c3b353c97edb0dd3d537260" FOREIGN KEY ("bot_id") REFERENCES "public"."bots"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP CONSTRAINT "FK_0a99c3b353c97edb0dd3d537260"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP CONSTRAINT "FK_40002c953ca3fde74084b8c2f6e"');
        await queryRunner.query('DROP INDEX "public"."IDX_0a99c3b353c97edb0dd3d53726"');
        await queryRunner.query('DROP INDEX "public"."IDX_40002c953ca3fde74084b8c2f6"');
        await queryRunner.query('DROP TABLE "public"."payonline_transactions"');
    }

}
