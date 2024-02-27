import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDPHONENUMBER1603439784238 implements MigrationInterface {
    name = 'ADDPHONENUMBER1603439784238'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."users" ADD "phone_number" character varying');
        await queryRunner.query('ALTER TABLE "public"."users" ADD CONSTRAINT "UQ_b53c008027e814b4bd95f3cf1bc" UNIQUE ("phone_number")');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."users" DROP CONSTRAINT "UQ_b53c008027e814b4bd95f3cf1bc"');
        await queryRunner.query('ALTER TABLE "public"."users" DROP COLUMN "phone_number"');
    }

}
