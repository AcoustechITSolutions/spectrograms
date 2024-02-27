import {MigrationInterface, QueryRunner} from 'typeorm';

export class AUTOGENERATE1614073626585 implements MigrationInterface {
    name = 'AUTOGENERATE1614073626585'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."user_subscriptions" ("id" SERIAL NOT NULL, "user_id" integer NOT NULL, "diagnostics_left" integer NOT NULL, CONSTRAINT "REL_7fb7e2469ce718e4205fb49004" UNIQUE ("user_id"), CONSTRAINT "PK_b64071df4bc7a2c4dd2da784c2b" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_7fb7e2469ce718e4205fb49004" ON "public"."user_subscriptions" ("user_id") ');
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum" RENAME TO "roles_role_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum" AS ENUM(\'patient\', \'doctor\', \'data_scientist\', \'admin\', \'external_server\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum" USING "role"::"text"::"public"."roles_role_enum"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum_old"');
        await queryRunner.query('ALTER TABLE "public"."user_subscriptions" ADD CONSTRAINT "FK_7fb7e2469ce718e4205fb490049" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."user_subscriptions" DROP CONSTRAINT "FK_7fb7e2469ce718e4205fb490049"');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum_old" AS ENUM(\'patient\', \'doctor\', \'data_scientist\', \'admin\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum_old" USING "role"::"text"::"public"."roles_role_enum_old"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum"');
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum_old" RENAME TO  "roles_role_enum"');
        await queryRunner.query('DROP INDEX "public"."IDX_7fb7e2469ce718e4205fb49004"');
        await queryRunner.query('DROP TABLE "public"."user_subscriptions"');
    }

}
