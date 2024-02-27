import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDDATASCIENTIST1600332062521 implements MigrationInterface {
    name = 'ADDDATASCIENTIST1600332062521'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26"');
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum" RENAME TO "roles_role_enum_old"');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum" AS ENUM(\'patient\', \'doctor\', \'data_scientist\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum" USING "role"::"text"::"public"."roles_role_enum"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum_old"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26" FOREIGN KEY ("intensity_id") REFERENCES "public"."cough_intensity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26"');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum_old" AS ENUM(\'patient\', \'doctor\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "public"."roles_role_enum_old" USING "role"::"text"::"public"."roles_role_enum_old"');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum"');
        await queryRunner.query('ALTER TYPE "public"."roles_role_enum_old" RENAME TO  "roles_role_enum"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26" FOREIGN KEY ("intensity_id") REFERENCES "public"."cough_intensity_types"("id") ON DELETE NO ACTION ON UPDATE CASCADE');
    }
}
