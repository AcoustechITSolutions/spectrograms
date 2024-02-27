import {MigrationInterface, QueryRunner} from "typeorm";

export class CODEVERIFIED1623076348772 implements MigrationInterface {
    name = 'CODEVERIFIED1623076348772'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" ADD "is_verified" boolean NOT NULL DEFAULT false`);
        await queryRunner.query(`ALTER TABLE "public"."users" ALTER COLUMN "password" DROP NOT NULL`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ALTER COLUMN "password" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" DROP COLUMN "is_verified"`);
    }

}
