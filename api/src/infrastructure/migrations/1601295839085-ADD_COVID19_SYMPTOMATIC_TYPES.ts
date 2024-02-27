import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDCOVID19SYMPTOMATICTYPES1601295839085 implements MigrationInterface {
    name = 'ADDCOVID19SYMPTOMATICTYPES1601295839085'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" RENAME COLUMN "is_covid19" TO "covid19_symptomatic_type_id"');
        await queryRunner.query('CREATE TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum" AS ENUM(\'no_covid19\', \'covid19_without_symptomatic\', \'covid19_mild_symptomatic\', \'covid19_severe_symptomatic\')');
        await queryRunner.query('CREATE TABLE "public"."covid19_symptomatic_types" ("id" SERIAL NOT NULL, "symptomatic_type" "public"."covid19_symptomatic_types_symptomatic_type_enum" NOT NULL, CONSTRAINT "UQ_c0017c628c0e43edbd8d4261302" UNIQUE ("symptomatic_type"), CONSTRAINT "PK_d33e9b4471f62bf489de8b8a023" PRIMARY KEY ("id"))');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" DROP COLUMN "covid19_symptomatic_type_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" ADD "covid19_symptomatic_type_id" integer');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" ADD CONSTRAINT "FK_6f3f91bd631e17577dcd665142f" FOREIGN KEY ("covid19_symptomatic_type_id") REFERENCES "public"."covid19_symptomatic_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" DROP CONSTRAINT "FK_6f3f91bd631e17577dcd665142f"');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" DROP COLUMN "covid19_symptomatic_type_id"');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" ADD "covid19_symptomatic_type_id" boolean');
        await queryRunner.query('DROP TABLE "public"."covid19_symptomatic_types"');
        await queryRunner.query('DROP TYPE "public"."covid19_symptomatic_types_symptomatic_type_enum"');
        await queryRunner.query('ALTER TABLE "public"."dataset_patient_diseases" RENAME COLUMN "covid19_symptomatic_type_id" TO "is_covid19"');
    }
}
