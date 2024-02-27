import {MigrationInterface, QueryRunner} from "typeorm";

export class NEWUSERROLES1616495597846 implements MigrationInterface {
    name = 'NEWUSERROLES1616495597846'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TYPE "public"."roles_role_enum" RENAME TO "roles_role_enum_old"`);
        await queryRunner.query(`CREATE TYPE "public"."roles_role_enum" AS ENUM('patient', 'dataset', 'edifier', 'doctor', 'data_scientist', 'admin', 'external_server')`);
        await queryRunner.query(`ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum" USING "role"::"text"::"public"."roles_role_enum"`);
        await queryRunner.query(`DROP TYPE "public"."roles_role_enum_old"`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TYPE "public"."roles_role_enum_old" AS ENUM('patient', 'doctor', 'data_scientist', 'admin', 'external_server')`);
        await queryRunner.query(`ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum_old" USING "role"::"text"::"public"."roles_role_enum_old"`);
        await queryRunner.query(`DROP TYPE "public"."roles_role_enum"`);
        await queryRunner.query(`ALTER TYPE "public"."roles_role_enum_old" RENAME TO  "roles_role_enum"`);
    }

}
