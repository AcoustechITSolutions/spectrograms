import {MigrationInterface, QueryRunner} from "typeorm";

export class VOICEEMBEDDING1632220718351 implements MigrationInterface {
    name = 'VOICEEMBEDDING1632220718351'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD "voice_embedding" character varying`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD "voice_audio_path" character varying`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP COLUMN "voice_audio_path"`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP COLUMN "voice_embedding"`);
    }

}
