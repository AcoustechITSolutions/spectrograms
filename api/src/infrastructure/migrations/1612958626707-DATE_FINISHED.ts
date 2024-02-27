import {MigrationInterface, QueryRunner} from 'typeorm';

export class DATEFINISHED1612958626707 implements MigrationInterface {
    name = 'DATEFINISHED1612958626707'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD "date_finished" TIMESTAMP WITH TIME ZONE');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD "date_finished" TIMESTAMP WITH TIME ZONE');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ALTER COLUMN "is_active" SET DEFAULT \'false\'');
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "date_time" DROP DEFAULT');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ALTER COLUMN "date_time" SET DEFAULT CURRENT_TIMESTAMP');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ALTER COLUMN "is_active" SET DEFAULT false');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP COLUMN "date_finished"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP COLUMN "date_finished"');
    }

}
