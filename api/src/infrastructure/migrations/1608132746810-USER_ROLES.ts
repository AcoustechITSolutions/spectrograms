import {MigrationInterface, QueryRunner} from 'typeorm';

export class USERROLES1608132746810 implements MigrationInterface {
    name = 'USERROLES1608132746810'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum" RENAME TO "roles_role_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum" AS ENUM(\'patient\', \'doctor\', \'data_scientist\', \'admin\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum" USING "role"::"text"::"public"."roles_role_enum"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum_old"');
        await queryRunner.query('COMMENT ON COLUMN "public"."roles"."role" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_dataset_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_diagnostic_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."tg_dataset_request"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."hw_diagnostic_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."dataset_request"."date_created" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."diagnostic_requests"."dateCreated" IS NULL');
        await queryRunner.query('COMMENT ON COLUMN "public"."roles"."role" IS NULL');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum_old" AS ENUM(\'patient\', \'doctor\', \'data_scientist\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum_old" USING "role"::"text"::"public"."roles_role_enum_old"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum"');
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum_old" RENAME TO  "roles_role_enum"');
    }

}
