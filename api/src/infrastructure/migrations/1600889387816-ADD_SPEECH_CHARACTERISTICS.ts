import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDSPEECHCHARACTERISTICS1600889387816 implements MigrationInterface {
    name = 'ADDSPEECHCHARACTERISTICS1600889387816'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."dataset_speech_characteristics" ("id" SERIAL NOT NULL, "request_id" integer NOT NULL, "commentary" character varying NOT NULL, CONSTRAINT "REL_c157b2053da77d33e17b3018af" UNIQUE ("request_id"), CONSTRAINT "PK_7c8002d2aab6c1af5d379ff6e93" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_c157b2053da77d33e17b3018af" ON "public"."dataset_speech_characteristics" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_speech_characteristics" ADD CONSTRAINT "FK_c157b2053da77d33e17b3018af8" FOREIGN KEY ("request_id") REFERENCES "public"."dataset_request"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_speech_characteristics" DROP CONSTRAINT "FK_c157b2053da77d33e17b3018af8"');
        await queryRunner.query('DROP INDEX "public"."IDX_c157b2053da77d33e17b3018af"');
        await queryRunner.query('DROP TABLE "public"."dataset_speech_characteristics"');
    }
}
