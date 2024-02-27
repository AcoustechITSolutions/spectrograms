import {MigrationInterface, QueryRunner} from "typeorm";

export class DIAGNOSTICSPECTROGRAM1637067557178 implements MigrationInterface {
    name = 'DIAGNOSTICSPECTROGRAM1637067557178'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."cough_audio" ADD "spectrogram_path" character varying(255)`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."cough_audio" DROP COLUMN "spectrogram_path"`);
    }

}
