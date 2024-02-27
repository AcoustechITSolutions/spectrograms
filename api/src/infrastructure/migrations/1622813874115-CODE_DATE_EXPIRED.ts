import {MigrationInterface, QueryRunner} from "typeorm";

export class CODEDATEEXPIRED1622813874115 implements MigrationInterface {
    name = 'CODEDATEEXPIRED1622813874115'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" ADD "date_expired" TIMESTAMP WITH TIME ZONE NOT NULL`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" DROP COLUMN "date_expired"`);
    }

}
