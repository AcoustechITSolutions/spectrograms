import {MigrationInterface, QueryRunner} from 'typeorm';

export class PAYMENTORDERID1613136863426 implements MigrationInterface {
    name = 'PAYMENTORDERID1613136863426'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "date_created" TIMESTAMP');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "request_id" integer');
        await queryRunner.query('CREATE INDEX "IDX_7729864d0e0c8953aa60cf853a" ON "public"."payonline_transactions" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD CONSTRAINT "FK_7729864d0e0c8953aa60cf853a2" FOREIGN KEY ("request_id") REFERENCES "public"."tg_new_diagnostic_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('UPDATE "public"."payonline_transactions" SET "date_created" = "date_time", "request_id" = "order_id"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "date_created" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "date_created" SET DEFAULT CURRENT_TIMESTAMP');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "request_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP CONSTRAINT "FK_40002c953ca3fde74084b8c2f6e"');
        await queryRunner.query('DROP INDEX "public"."IDX_40002c953ca3fde74084b8c2f6"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "date_time"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "order_id"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "date_updated" TIMESTAMP');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "transaction_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "security_key" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "is_confirmed" SET DEFAULT false');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "date_time" TIMESTAMP');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "order_id" integer'); 
        await queryRunner.query('CREATE INDEX "IDX_40002c953ca3fde74084b8c2f6" ON "public"."payonline_transactions" ("order_id") ');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD CONSTRAINT "FK_40002c953ca3fde74084b8c2f6e" FOREIGN KEY ("order_id") REFERENCES "public"."tg_new_diagnostic_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('UPDATE "public"."payonline_transactions" SET "date_time" = "date_created", "order_id" = "request_id"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "order_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "date_time" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "date_created"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "request_id"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP CONSTRAINT "FK_7729864d0e0c8953aa60cf853a2"');
        await queryRunner.query('DROP INDEX "public"."IDX_7729864d0e0c8953aa60cf853a"');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "is_confirmed" DROP DEFAULT');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "security_key" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "transaction_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "date_updated"');
    }

}
