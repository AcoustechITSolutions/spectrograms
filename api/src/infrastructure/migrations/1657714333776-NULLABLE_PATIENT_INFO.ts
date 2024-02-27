import {MigrationInterface, QueryRunner} from "typeorm";

export class NULLABLEPATIENTINFO1657714333776 implements MigrationInterface {
    name = 'NULLABLEPATIENTINFO1657714333776'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_70a6625ea9db5e2cc31d40c191c"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_b80985f2b28eb64cc33da447ad4"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_ee58734eb129a019ada294d453b"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP COLUMN "disease_type_id"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP COLUMN "acute_cough_types_id"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP COLUMN "chronic_cough_types_id"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" DROP COLUMN "privacy_eula_version"`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "age" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "is_smoking" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "is_forced" DROP NOT NULL`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "is_forced" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "is_smoking" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ALTER COLUMN "age" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD "privacy_eula_version" integer`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD "chronic_cough_types_id" integer`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD "acute_cough_types_id" integer`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD "disease_type_id" integer`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_ee58734eb129a019ada294d453b" FOREIGN KEY ("chronic_cough_types_id") REFERENCES "public"."chronic_cough_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_b80985f2b28eb64cc33da447ad4" FOREIGN KEY ("acute_cough_types_id") REFERENCES "public"."acute_cough_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_70a6625ea9db5e2cc31d40c191c" FOREIGN KEY ("disease_type_id") REFERENCES "public"."disease_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

}
