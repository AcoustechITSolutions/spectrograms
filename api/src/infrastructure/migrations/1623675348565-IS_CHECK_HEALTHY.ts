import {MigrationInterface, QueryRunner} from "typeorm";

export class ISCHECKHEALTHY1623675348565 implements MigrationInterface {
    name = 'ISCHECKHEALTHY1623675348565'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "is_check_healthy" boolean NOT NULL DEFAULT false`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "is_check_healthy"`);
    }

}
