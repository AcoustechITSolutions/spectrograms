import {MigrationInterface, QueryRunner} from "typeorm";

export class CHECKINGINTERVAL1644513227253 implements MigrationInterface {
    name = 'CHECKINGINTERVAL1644513227253'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "check_start" character varying`);
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "check_end" character varying`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "check_end"`);
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "check_start"`);
    }

}
