import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXSPEECHCOMMENTARY1605178712632 implements MigrationInterface {
    name = 'FIXSPEECHCOMMENTARY1605178712632'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_speech_characteristics" ALTER COLUMN "commentary" SET DEFAULT \'\'');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_speech_characteristics" ALTER COLUMN "commentary" DROP DEFAULT');
    }

}
