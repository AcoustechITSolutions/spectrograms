import {MigrationInterface, QueryRunner} from "typeorm";

export class USEREMAILERROR1622707400744 implements MigrationInterface {
    name = 'USEREMAILERROR1622707400744'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "is_email_error" boolean NOT NULL DEFAULT false`);
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "email_error_type" character varying`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "email_error_type"`);
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "is_email_error"`);
    }

}
