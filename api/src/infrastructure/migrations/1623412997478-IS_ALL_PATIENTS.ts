import {MigrationInterface, QueryRunner} from "typeorm";

export class ISALLPATIENTS1623412997478 implements MigrationInterface {
    name = 'ISALLPATIENTS1623412997478'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "is_all_patients" boolean NOT NULL DEFAULT true`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "is_all_patients"`);
    }

}
