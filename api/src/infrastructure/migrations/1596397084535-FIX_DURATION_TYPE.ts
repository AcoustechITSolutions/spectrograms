import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXDURATIONTYPE1596397084535 implements MigrationInterface {
    name = 'FIXDURATIONTYPE1596397084535'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_audio" ALTER COLUMN "duration" set data type double precision');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_audio" ALTER COLUMN "duration" set data type integer');
    }
}
