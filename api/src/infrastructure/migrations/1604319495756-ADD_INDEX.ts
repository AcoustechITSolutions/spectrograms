import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDINDEX1604319495756 implements MigrationInterface {
    name = 'ADDINDEX1604319495756'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE INDEX "IDX_39de5cf907cced3bc6910fca2d" ON "public"."tg_diagnostic_request" ("request_id") ');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('DROP INDEX "public"."IDX_39de5cf907cced3bc6910fca2d"');
    }

}
