import {MigrationInterface, QueryRunner} from "typeorm";

export class ISVALIDATECOUGH1628162118633 implements MigrationInterface {
    name = 'ISVALIDATECOUGH1628162118633'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" ADD "is_validate_cough" boolean NOT NULL DEFAULT true`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."users" DROP COLUMN "is_validate_cough"`);
    }

}
