import {MigrationInterface, QueryRunner} from "typeorm";

export class NULLABLEGENDER1634138457159 implements MigrationInterface {
    name = 'NULLABLEGENDER1634138457159'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "gender" DROP NOT NULL`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "gender" SET NOT NULL`);
    }

}
