import {MigrationInterface, QueryRunner} from 'typeorm';

export class OTHERPAYMENT1611223178438 implements MigrationInterface {
    name = 'OTHERPAYMENT1611223178438'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."bot_payments" DROP COLUMN "payment_sum"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" ADD "payment_sum" integer NOT NULL DEFAULT \'0\'');
        await queryRunner.query('ALTER TYPE "public"."payment_types_payment_type_enum" RENAME TO "payment_types_payment_type_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."payment_types_payment_type_enum" AS ENUM(\'free\', \'single\', \'unlimited\', \'other\')');
        await queryRunner.query('ALTER TABLE "public"."payment_types" ALTER COLUMN "payment_type" TYPE "public"."payment_types_payment_type_enum" USING "payment_type"::"text"::"public"."payment_types_payment_type_enum"');
        await queryRunner.query('DROP TYPE "public"."payment_types_payment_type_enum_old"');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ALTER COLUMN "is_active" SET DEFAULT \'false\'');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ALTER COLUMN "is_active" SET DEFAULT false');
        await queryRunner.query('CREATE TYPE "public"."payment_types_payment_type_enum_old" AS ENUM(\'single\', \'unlimited\')');
        await queryRunner.query('ALTER TABLE "public"."payment_types" ALTER COLUMN "payment_type" TYPE "public"."payment_types_payment_type_enum_old" USING "payment_type"::"text"::"public"."payment_types_payment_type_enum_old"');
        await queryRunner.query('DROP TYPE "public"."payment_types_payment_type_enum"');
        await queryRunner.query('ALTER TYPE "public"."payment_types_payment_type_enum_old" RENAME TO  "payment_types_payment_type_enum"');
        await queryRunner.query('ALTER TABLE "public"."tg_new_diagnostic_request" DROP COLUMN "payment_sum"');
        await queryRunner.query('ALTER TABLE "public"."bot_payments" ADD "payment_sum" integer');
    }

}
