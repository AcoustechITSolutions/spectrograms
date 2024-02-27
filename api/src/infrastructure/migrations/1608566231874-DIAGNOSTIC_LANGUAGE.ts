import {MigrationInterface, QueryRunner} from 'typeorm';

export class DIAGNOSTICLANGUAGE1608566231874 implements MigrationInterface {
    name = 'DIAGNOSTICLANGUAGE1608566231874'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" ADD "language" character varying');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" DROP COLUMN "language"');
    }

}
