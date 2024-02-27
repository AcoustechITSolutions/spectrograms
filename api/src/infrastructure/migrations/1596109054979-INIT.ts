import {MigrationInterface, QueryRunner} from 'typeorm';
// await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" TYPE "roles_role_enum" USING CASE WHEN role == 0 THEN "patient" ELSE "doctor" END');
export class INIT1596109054979 implements MigrationInterface {
    name = 'INIT1596109054979'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."roles" ("id" SERIAL NOT NULL, "role" integer NOT NULL, CONSTRAINT "UQ_a42a2b3f8b1da89cca94267d7d1" UNIQUE ("role"), CONSTRAINT "PK_130f0eec948cd435a779de3a4f0" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."users" ("id" SERIAL NOT NULL, "login" character varying(255) NOT NULL, "email" character varying(255) NOT NULL, "password" character varying(255) NOT NULL, CONSTRAINT "UQ_32ab990ec2829bffa04c63e6f8d" UNIQUE ("login"), CONSTRAINT "UQ_12ffa5c867f6bb71e2690a526ce" UNIQUE ("email"), CONSTRAINT "PK_a6cc71bedf15a41a5f5ee8aea97" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."users_roles_roles" ("usersId" integer NOT NULL, "rolesId" integer NOT NULL, CONSTRAINT "PK_3ded5a3da59c96dcc6bfcb58def" PRIMARY KEY ("usersId", "rolesId"))');
        await queryRunner.query('CREATE INDEX "IDX_0c8e8f7c8d3f324ddda2f1d999" ON "public"."users_roles_roles" ("usersId") ');
        await queryRunner.query('CREATE INDEX "IDX_3d34e95c3fc3acab78e7184fc1" ON "public"."users_roles_roles" ("rolesId") ');
        await queryRunner.query('ALTER TABLE "public"."users_roles_roles" ADD CONSTRAINT "FK_0c8e8f7c8d3f324ddda2f1d999b" FOREIGN KEY ("usersId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."users_roles_roles" ADD CONSTRAINT "FK_3d34e95c3fc3acab78e7184fc19" FOREIGN KEY ("rolesId") REFERENCES "public"."roles"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."users_roles_roles" DROP CONSTRAINT "FK_3d34e95c3fc3acab78e7184fc19"');
        await queryRunner.query('ALTER TABLE "public"."users_roles_roles" DROP CONSTRAINT "FK_0c8e8f7c8d3f324ddda2f1d999b"');
        await queryRunner.query('DROP INDEX "public"."IDX_3d34e95c3fc3acab78e7184fc1"');
        await queryRunner.query('DROP INDEX "public"."IDX_0c8e8f7c8d3f324ddda2f1d999"');
        await queryRunner.query('DROP TABLE "public"."users_roles_roles"');
        await queryRunner.query('DROP TABLE "public"."users"');
        await queryRunner.query('DROP TABLE "public"."roles"');
    }
}
