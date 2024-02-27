import {MigrationInterface, QueryRunner} from "typeorm";

export class QRTOKENUUID1627289958055 implements MigrationInterface {
    name = 'QRTOKENUUID1627289958055'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" ADD "qr_code_token" uuid NOT NULL DEFAULT gen_random_uuid()`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_report" DROP COLUMN "qr_code_token"`);
    }

}
