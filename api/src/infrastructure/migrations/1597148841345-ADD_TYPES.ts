import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDTYPES1597148841345 implements MigrationInterface {
    name = 'ADDTYPES1597148841345'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."acute_cough_types_acute_cough_types_enum" AS ENUM(\'acute_bronchitis\', \'viral_pneumonia\', \'pleurisy\', \'pulmonary_embolism\', \'whooping_cough\', \'pneumonia\', \'pneumonia_complication\', \'other\')');
        await queryRunner.query('CREATE TABLE "public"."acute_cough_types" ("id" SERIAL NOT NULL, "acute_cough_types" "public"."acute_cough_types_acute_cough_types_enum" NOT NULL, CONSTRAINT "UQ_85bba60b78f3cd5cd2fcf370dea" UNIQUE ("acute_cough_types"), CONSTRAINT "PK_104e982b8984ba6c7931504430a" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."chronic_cough_types_chronic_cough_type_enum" AS ENUM(\'lung_infarction\', \'bronchial_asthma\', \'psychogenic_cough\', \'primary_tuberculosis_complex\', \'chronical_bronchitis\', \'copd\', \'bronchoectatic_disease\', \'tumors\', \'congestive_heart_failure\', \'other\')');
        await queryRunner.query('CREATE TABLE "public"."chronic_cough_types" ("id" SERIAL NOT NULL, "chronic_cough_type" "public"."chronic_cough_types_chronic_cough_type_enum" NOT NULL, CONSTRAINT "UQ_757e6adba9e41fd673088d438b2" UNIQUE ("chronic_cough_type"), CONSTRAINT "PK_107d5e544d81001ca526595746b" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."disease_types_disease_type_enum" AS ENUM(\'acute\', \'chronic\', \'none\')');
        await queryRunner.query('CREATE TABLE "public"."disease_types" ("id" SERIAL NOT NULL, "disease_type" "public"."disease_types_disease_type_enum" NOT NULL, CONSTRAINT "UQ_60e99305113b1d1dc4caae7f943" UNIQUE ("disease_type"), CONSTRAINT "PK_7f84dd91701b1082d28c3a2c29d" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."patient_info_acute_cough_types_acute_cough_types" ("patientInfoId" integer NOT NULL, "acuteCoughTypesId" integer NOT NULL, CONSTRAINT "PK_3e2ff34a2debb8568e1012eb035" PRIMARY KEY ("patientInfoId", "acuteCoughTypesId"))');
        await queryRunner.query('CREATE INDEX "IDX_59e644027ce868d7f24dca4ea0" ON "public"."patient_info_acute_cough_types_acute_cough_types" ("patientInfoId") ');
        await queryRunner.query('CREATE INDEX "IDX_7020ecbbc43dc921f39aad2552" ON "public"."patient_info_acute_cough_types_acute_cough_types" ("acuteCoughTypesId") ');
        await queryRunner.query('CREATE TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types" ("patientInfoId" integer NOT NULL, "chronicCoughTypesId" integer NOT NULL, CONSTRAINT "PK_e5e81b8897cadb83ea5c21381f2" PRIMARY KEY ("patientInfoId", "chronicCoughTypesId"))');
        await queryRunner.query('CREATE INDEX "IDX_ddbc76190e28ba0f66d7a69444" ON "public"."patient_info_chronic_cough_types_chronic_cough_types" ("patientInfoId") ');
        await queryRunner.query('CREATE INDEX "IDX_64f2609a853e46a88fe6f4634c" ON "public"."patient_info_chronic_cough_types_chronic_cough_types" ("chronicCoughTypesId") ');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD "disease_type_id" integer');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_70a6625ea9db5e2cc31d40c191c" FOREIGN KEY ("disease_type_id") REFERENCES "public"."disease_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info_acute_cough_types_acute_cough_types" ADD CONSTRAINT "FK_59e644027ce868d7f24dca4ea06" FOREIGN KEY ("patientInfoId") REFERENCES "public"."patient_info"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info_acute_cough_types_acute_cough_types" ADD CONSTRAINT "FK_7020ecbbc43dc921f39aad25525" FOREIGN KEY ("acuteCoughTypesId") REFERENCES "public"."acute_cough_types"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types" ADD CONSTRAINT "FK_ddbc76190e28ba0f66d7a69444a" FOREIGN KEY ("patientInfoId") REFERENCES "public"."patient_info"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types" ADD CONSTRAINT "FK_64f2609a853e46a88fe6f4634c3" FOREIGN KEY ("chronicCoughTypesId") REFERENCES "public"."chronic_cough_types"("id") ON DELETE CASCADE ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types" DROP CONSTRAINT "FK_64f2609a853e46a88fe6f4634c3"');
        await queryRunner.query('ALTER TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types" DROP CONSTRAINT "FK_ddbc76190e28ba0f66d7a69444a"');
        await queryRunner.query('ALTER TABLE "public"."patient_info_acute_cough_types_acute_cough_types" DROP CONSTRAINT "FK_7020ecbbc43dc921f39aad25525"');
        await queryRunner.query('ALTER TABLE "public"."patient_info_acute_cough_types_acute_cough_types" DROP CONSTRAINT "FK_59e644027ce868d7f24dca4ea06"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_70a6625ea9db5e2cc31d40c191c"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "UQ_70a6625ea9db5e2cc31d40c191c"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP COLUMN "disease_type_id"');
        await queryRunner.query('DROP INDEX "public"."IDX_64f2609a853e46a88fe6f4634c"');
        await queryRunner.query('DROP INDEX "public"."IDX_ddbc76190e28ba0f66d7a69444"');
        await queryRunner.query('DROP TABLE "public"."patient_info_chronic_cough_types_chronic_cough_types"');
        await queryRunner.query('DROP INDEX "public"."IDX_7020ecbbc43dc921f39aad2552"');
        await queryRunner.query('DROP INDEX "public"."IDX_59e644027ce868d7f24dca4ea0"');
        await queryRunner.query('DROP TABLE "public"."patient_info_acute_cough_types_acute_cough_types"');
        await queryRunner.query('DROP TABLE "public"."disease_types"');
        await queryRunner.query('DROP TYPE "public"."disease_types_disease_type_enum"');
        await queryRunner.query('DROP TABLE "public"."chronic_cough_types"');
        await queryRunner.query('DROP TYPE "public"."chronic_cough_types_chronic_cough_type_enum"');
        await queryRunner.query('DROP TABLE "public"."acute_cough_types"');
        await queryRunner.query('DROP TYPE "public"."acute_cough_types_acute_cough_types_enum"');
    }
}
