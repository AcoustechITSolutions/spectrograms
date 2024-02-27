import {MigrationInterface, QueryRunner} from 'typeorm';

export class NOSECURITYKEY1613144022708 implements MigrationInterface {
    name = 'NOSECURITYKEY1613144022708'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" DROP COLUMN "security_key"');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."payonline_transactions" ADD "security_key" character varying');
    }

}
