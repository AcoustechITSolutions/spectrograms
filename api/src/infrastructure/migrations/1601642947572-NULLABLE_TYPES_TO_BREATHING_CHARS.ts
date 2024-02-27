import {MigrationInterface, QueryRunner} from 'typeorm';

export class NULLABLETYPESTOBREATHINGCHARS1601642947572 implements MigrationInterface {
    name = 'NULLABLETYPESTOBREATHINGCHARS1601642947572'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_50201d631c0f1c9881421fdb91c"');
        await queryRunner.query('DROP INDEX "public"."IDX_13a303196f467042f0c28dccd1"');
        await queryRunner.query('DROP INDEX "public"."IDX_81adbb63d1cd6f90c1f2be43c4"');
        await queryRunner.query('DROP INDEX "public"."IDX_6b898cccddec97bac65b0c8249"');
        await queryRunner.query('DROP INDEX "public"."IDX_f9af6bf015afea3b963acfd9ba"');
        await queryRunner.query('DROP INDEX "public"."IDX_e65a9a77bd695eabb5f15bd01c"');
        await queryRunner.query('DROP INDEX "public"."IDX_6d459e887a9fe78766ae6b6c54"');
        await queryRunner.query('DROP INDEX "public"."IDX_357b4438d60c4ae6bb36a1772c"');
        await queryRunner.query('DROP INDEX "public"."IDX_0d893a2b196cd313cbc191cd62"');
        await queryRunner.query('DROP INDEX "public"."IDX_50201d631c0f1c9881421fdb91"');
        await queryRunner.query('DROP INDEX "public"."IDX_fc51b6792323b54950393b076c"');
        await queryRunner.query('DROP INDEX "public"."IDX_e8fa9a5ea426ac5cde060bde25"');
        await queryRunner.query('DROP INDEX "public"."IDX_a28a0967afe063e205f9c758f4"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_81adbb63d1cd6f90c1f2be43c46"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_6b898cccddec97bac65b0c82492"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_f9af6bf015afea3b963acfd9bab"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "depth_type_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "difficulty_type_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "duration_type_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_fc51b6792323b54950393b076c5"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ALTER COLUMN "marking_status_id" DROP DEFAULT');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ALTER COLUMN "doctor_status_id" DROP DEFAULT');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_81adbb63d1cd6f90c1f2be43c46" FOREIGN KEY ("depth_type_id") REFERENCES "public"."breathing_depth_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_6b898cccddec97bac65b0c82492" FOREIGN KEY ("difficulty_type_id") REFERENCES "public"."breathing_difficulty_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_f9af6bf015afea3b963acfd9bab" FOREIGN KEY ("duration_type_id") REFERENCES "public"."breathing_duration_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_50201d631c0f1c9881421fdb91c" FOREIGN KEY ("status_id") REFERENCES "public"."dataset_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_fc51b6792323b54950393b076c5" FOREIGN KEY ("marking_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254" FOREIGN KEY ("doctor_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_fc51b6792323b54950393b076c5"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" DROP CONSTRAINT "FK_50201d631c0f1c9881421fdb91c"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_f9af6bf015afea3b963acfd9bab"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_6b898cccddec97bac65b0c82492"');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" DROP CONSTRAINT "FK_81adbb63d1cd6f90c1f2be43c46"');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ALTER COLUMN "doctor_status_id" SET DEFAULT 2');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ALTER COLUMN "marking_status_id" SET DEFAULT 2');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_e8fa9a5ea426ac5cde060bde254" FOREIGN KEY ("doctor_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_fc51b6792323b54950393b076c5" FOREIGN KEY ("marking_status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "duration_type_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "difficulty_type_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ALTER COLUMN "depth_type_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_f9af6bf015afea3b963acfd9bab" FOREIGN KEY ("duration_type_id") REFERENCES "public"."breathing_duration_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_6b898cccddec97bac65b0c82492" FOREIGN KEY ("difficulty_type_id") REFERENCES "public"."breathing_difficulty_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."dataset_breathing_characteristics" ADD CONSTRAINT "FK_81adbb63d1cd6f90c1f2be43c46" FOREIGN KEY ("depth_type_id") REFERENCES "public"."breathing_depth_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('CREATE INDEX "IDX_a28a0967afe063e205f9c758f4" ON "public"."dataset_audio_info" ("audio_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_e8fa9a5ea426ac5cde060bde25" ON "public"."dataset_request" ("doctor_status_id") ');
        await queryRunner.query('CREATE INDEX "IDX_fc51b6792323b54950393b076c" ON "public"."dataset_request" ("marking_status_id") ');
        await queryRunner.query('CREATE INDEX "IDX_50201d631c0f1c9881421fdb91" ON "public"."dataset_request" ("status_id") ');
        await queryRunner.query('CREATE INDEX "IDX_0d893a2b196cd313cbc191cd62" ON "public"."dataset_patient_diseases" ("chronic_cough_types_id") ');
        await queryRunner.query('CREATE INDEX "IDX_357b4438d60c4ae6bb36a1772c" ON "public"."dataset_patient_diseases" ("acute_cough_types_id") ');
        await queryRunner.query('CREATE INDEX "IDX_6d459e887a9fe78766ae6b6c54" ON "public"."dataset_patient_diseases" ("disease_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_e65a9a77bd695eabb5f15bd01c" ON "public"."dataset_patient_details" ("gender_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_f9af6bf015afea3b963acfd9ba" ON "public"."dataset_breathing_characteristics" ("duration_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_6b898cccddec97bac65b0c8249" ON "public"."dataset_breathing_characteristics" ("difficulty_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_81adbb63d1cd6f90c1f2be43c4" ON "public"."dataset_breathing_characteristics" ("depth_type_id") ');
        await queryRunner.query('CREATE INDEX "IDX_13a303196f467042f0c28dccd1" ON "public"."dataset_breathing_characteristics" ("breathing_type_id") ');
        await queryRunner.query('ALTER TABLE "public"."dataset_request" ADD CONSTRAINT "FK_50201d631c0f1c9881421fdb91c" FOREIGN KEY ("status_id") REFERENCES "public"."dataset_marking_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
