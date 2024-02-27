import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDSPEECH1598093474210 implements MigrationInterface {
    name = 'ADDSPEECH1598093474210'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."speech_audio" ("id" SERIAL NOT NULL, "file_path" character varying NOT NULL, "request_id" integer NOT NULL, CONSTRAINT "REL_59a848060676de2b80b9978eb3" UNIQUE ("request_id"), CONSTRAINT "PK_fd1e05aaa20757e33245cc4b1cd" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."speech_audio" ADD CONSTRAINT "FK_59a848060676de2b80b9978eb38" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."speech_audio" DROP CONSTRAINT "FK_59a848060676de2b80b9978eb38"');
        await queryRunner.query('DROP TABLE "public"."speech_audio"');
    }
}
