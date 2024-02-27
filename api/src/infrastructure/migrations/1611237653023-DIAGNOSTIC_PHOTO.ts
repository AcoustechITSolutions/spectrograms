import {MigrationInterface, QueryRunner} from 'typeorm';

export class DIAGNOSTICPHOTO1611237653023 implements MigrationInterface {
    name = 'DIAGNOSTICPHOTO1611237653023'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "photo_path" character varying(255)');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "photo_path"');
    }

}
