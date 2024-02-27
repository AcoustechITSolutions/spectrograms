import {MigrationInterface, QueryRunner} from 'typeorm';

export class USERADDIsActive1612896378836 implements MigrationInterface {
    name = 'USERADDIsActive1612896378836'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE EXTENSION IF NOT EXISTS pgcrypto');
        await queryRunner.query('ALTER TABLE "public"."users" ADD "is_active" boolean NOT NULL DEFAULT true');
        await queryRunner.query('CREATE TABLE "public"."refresh_token" ("id" uuid NOT NULL DEFAULT gen_random_uuid(), "jwt_id" character varying(255) NOT NULL, "expires_date" TIMESTAMP NOT NULL, "created_date" TIMESTAMP NOT NULL DEFAULT now(), "user_id" integer, CONSTRAINT "UQ_c575679f1943fffa37d76214ffc" UNIQUE ("jwt_id"), CONSTRAINT "PK_d8468f7e763158b5bb1d06b929a" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_7820dc655b460aa1c381ac1494" ON "public"."refresh_token" ("user_id") ');
        await queryRunner.query('ALTER TABLE "public"."refresh_token" ADD CONSTRAINT "FK_7820dc655b460aa1c381ac14949" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."refresh_token" DROP CONSTRAINT "FK_7820dc655b460aa1c381ac14949"');
        await queryRunner.query('DROP INDEX "public"."IDX_7820dc655b460aa1c381ac1494"');
        await queryRunner.query('DROP TABLE "public"."refresh_token"');
        await queryRunner.query('ALTER TABLE "public"."users" DROP COLUMN "is_active"');
    }
}
