import {MigrationInterface, QueryRunner} from "typeorm";

export class ISCHECKCOVID1626694785692 implements MigrationInterface {
    name = 'ISCHECKCOVID1626694785692'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "is_check_covid" boolean NOT NULL DEFAULT true`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "is_check_covid"`);
    }

}
