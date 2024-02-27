import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDREPRESENTATIVEANDBREATHINGCOMMENTARY1602680735798 implements MigrationInterface {
    name = 'ADDREPRESENTATIVEANDBREATHINGCOMMENTARY1602680735798'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."dataset_breathing_general_info" ("id" SERIAL NOT NULL, "request_id" integer NOT NULL, "commentary" character varying, CONSTRAINT "REL_98e2c7dad92460f3da6e35be12" UNIQUE ("request_id"), CONSTRAINT "PK_899fde67d1b61d456a894b7ad84" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_98e2c7dad92460f3da6e35be12" ON "public"."dataset_breathing_general_info" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" ADD "is_representative" boolean');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_general_info" ADD CONSTRAINT "FK_98e2c7dad92460f3da6e35be12c" FOREIGN KEY ("request_id") REFERENCES "public"."dataset_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_general_info" DROP CONSTRAINT "FK_98e2c7dad92460f3da6e35be12c"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "is_representative"');
        await queryRunner.query('DROP INDEX "public"."IDX_98e2c7dad92460f3da6e35be12"');
        await queryRunner.query('DROP TABLE "public"."dataset_breathing_general_info"');
    }

}
