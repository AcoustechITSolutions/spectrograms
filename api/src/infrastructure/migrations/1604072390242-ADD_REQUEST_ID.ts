import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDREQUESTID1604072390242 implements MigrationInterface {
    name = 'ADDREQUESTID1604072390242'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_audio" DROP COLUMN "uuid"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD "request_id" integer');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD CONSTRAINT "UQ_39de5cf907cced3bc6910fca2d0" UNIQUE ("request_id")');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" ADD CONSTRAINT "FK_39de5cf907cced3bc6910fca2d0" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP CONSTRAINT "FK_39de5cf907cced3bc6910fca2d0"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP CONSTRAINT "UQ_39de5cf907cced3bc6910fca2d0"');
        await queryRunner.query('ALTER TABLE "public"."tg_diagnostic_request" DROP COLUMN "request_id"');
        await queryRunner.query('ALTER TABLE "public"."cough_audio" ADD "uuid" character varying NOT NULL');
    }

}
