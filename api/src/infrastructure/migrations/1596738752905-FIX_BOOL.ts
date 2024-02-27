import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXBOOL1596738752905 implements MigrationInterface {
    name = 'FIXBOOL1596738752905'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_user_visible" DROP DEFAULT');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "is_user_visible" SET DEFAULT true');
    }
}
