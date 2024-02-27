import {MigrationInterface, QueryRunner} from 'typeorm';

export class PATIENTINFOFIXES1597412463218 implements MigrationInterface {
    name = 'PATIENTINFOFIXES1597412463218'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "identifier" character varying');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "sick_days" SET NOT NULL');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "sick_days" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "identifier"');
    }
}
