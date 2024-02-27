import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDDIAGNOSTIC1596389223721 implements MigrationInterface {
    name = 'ADDDIAGNOSTIC1596389223721'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."diagnostic_request_status_request_status_enum" AS ENUM(\'pending\', \'error\', \'processing\', \'success\')');
        await queryRunner.query('CREATE TABLE "public"."diagnostic_request_status" ("id" SERIAL NOT NULL, "request_status" "public"."diagnostic_request_status_request_status_enum" NOT NULL, CONSTRAINT "PK_5ef837af869e4c668e5dfb7f91e" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."diagnostic_requests" ("id" SERIAL NOT NULL, "user_id" integer NOT NULL, "status_id" integer NOT NULL, "dateCreated" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, CONSTRAINT "PK_c5cc54c18225a84d8e33db1c6e8" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_8239e9f19c1aedde804a20bd0b" ON "public"."diagnostic_requests" ("user_id") ');
        await queryRunner.query('CREATE INDEX "IDX_a9e0f73503ee7274446131da1b" ON "public"."diagnostic_requests" ("status_id") ');
        await queryRunner.query('CREATE TABLE "public"."cough_audio" ("id" SERIAL NOT NULL, "file_path" character varying(255) NOT NULL, "request_id" integer NOT NULL, "samplerate" integer, "duration" integer, CONSTRAINT "REL_9aa172194f6e5fcbba13925f07" UNIQUE ("request_id"), CONSTRAINT "PK_b850d79e214d373e3fa58f0ae36" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."cough_intensity_types_intensity_type_enum" AS ENUM(\'intensive\', \'not_intensive\', \'weak\')');
        await queryRunner.query('CREATE TABLE "public"."cough_intensity_types" ("id" SERIAL NOT NULL, "intensity_type" "public"."cough_intensity_types_intensity_type_enum" NOT NULL, CONSTRAINT "UQ_1591d461a43630c4354501ea0c7" UNIQUE ("intensity_type"), CONSTRAINT "PK_c1645cbda1bb88f1a728afc5aa7" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."cough_productivity_types_productivity_type_enum" AS ENUM(\'productive\', \'unproductive\')');
        await queryRunner.query('CREATE TABLE "public"."cough_productivity_types" ("id" SERIAL NOT NULL, "productivity_type" "public"."cough_productivity_types_productivity_type_enum" NOT NULL, CONSTRAINT "UQ_5e1f331517d1baca1b5b26a54a1" UNIQUE ("productivity_type"), CONSTRAINT "PK_35a72f4ea05e66b25b1f41b5114" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."cough_transitivity_types_transitivity_type_enum" AS ENUM(\'wet_productive_small\', \'dry_productive_small\', \'intransitive\')');
        await queryRunner.query('CREATE TABLE "public"."cough_transitivity_types" ("id" SERIAL NOT NULL, "transitivity_type" "public"."cough_transitivity_types_transitivity_type_enum" NOT NULL, CONSTRAINT "UQ_9a99d86618271b8714e1b5a7cbb" UNIQUE ("transitivity_type"), CONSTRAINT "PK_86deb7c17bac8b4eb19a1954c78" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."cough_characteristics" ("id" SERIAL NOT NULL, "is_forced" boolean NOT NULL, "productivity_id" integer NOT NULL, "transitivity_id" integer NOT NULL, "intensity_id" integer NOT NULL, "requestId" integer, CONSTRAINT "REL_dd304f9a097ae06778a2bab3e5" UNIQUE ("requestId"), CONSTRAINT "PK_523b77c1c4e20e94b728319f4ac" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TABLE "public"."diagnostic_report" ("id" SERIAL NOT NULL, "user_id" integer NOT NULL, "report_path" character varying(255) NOT NULL, "commentary" character varying NOT NULL, "is_confirmed" boolean NOT NULL, "requestId" integer, CONSTRAINT "REL_4889b4fb25a2db3cd74ddc51ad" UNIQUE ("requestId"), CONSTRAINT "PK_fb5edf7de49f4c190baaab2dbef" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_4889b4fb25a2db3cd74ddc51ad" ON "public"."diagnostic_report" ("requestId") ');
        await queryRunner.query('CREATE INDEX "IDX_a8398b11b6e9516e92c8c96b92" ON "public"."diagnostic_report" ("user_id") ');
        await queryRunner.query('CREATE TYPE "public"."patient_info_gender_enum" AS ENUM(\'male\', \'female\')');
        await queryRunner.query('CREATE TABLE "public"."patient_info" ("id" SERIAL NOT NULL, "age" integer NOT NULL, "gender" "public"."patient_info_gender_enum" NOT NULL, "is_smoking" boolean NOT NULL, "sick_days" integer NOT NULL, "requestId" integer, CONSTRAINT "REL_81031c936ec7d07cbea5d347ad" UNIQUE ("requestId"), CONSTRAINT "PK_683a79938ad0edcf673c330c2e3" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE TYPE "public"."roles_role_enum" AS ENUM(\'patient\', \'doctor\')');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" set data type roles_role_enum using case when role = 0 then \'patient\'::roles_role_enum else \'doctor\'::roles_role_enum END');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" ADD CONSTRAINT "FK_8239e9f19c1aedde804a20bd0bd" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" ADD CONSTRAINT "FK_a9e0f73503ee7274446131da1b0" FOREIGN KEY ("status_id") REFERENCES "public"."diagnostic_request_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_audio" ADD CONSTRAINT "FK_9aa172194f6e5fcbba13925f07d" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dd304f9a097ae06778a2bab3e5b" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541" FOREIGN KEY ("productivity_id") REFERENCES "public"."cough_productivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb" FOREIGN KEY ("transitivity_id") REFERENCES "public"."cough_transitivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26" FOREIGN KEY ("intensity_id") REFERENCES "public"."cough_intensity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD CONSTRAINT "FK_4889b4fb25a2db3cd74ddc51ad0" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD CONSTRAINT "FK_a8398b11b6e9516e92c8c96b92b" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_81031c936ec7d07cbea5d347adc" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_81031c936ec7d07cbea5d347adc"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP CONSTRAINT "FK_a8398b11b6e9516e92c8c96b92b"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP CONSTRAINT "FK_4889b4fb25a2db3cd74ddc51ad0"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dd304f9a097ae06778a2bab3e5b"');
        await queryRunner.query('ALTER TABLE "public"."cough_audio" DROP CONSTRAINT "FK_9aa172194f6e5fcbba13925f07d"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" DROP CONSTRAINT "FK_a9e0f73503ee7274446131da1b0"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_requests" DROP CONSTRAINT "FK_8239e9f19c1aedde804a20bd0bd"');
        await queryRunner.query('ALTER TABLE "public"."roles" ALTER COLUMN "role" set data type integer using case when role = \'patient\'::roles_role_enum then 0 else 1 END');
        await queryRunner.query('DROP TYPE "public"."roles_role_enum"');
        await queryRunner.query('DROP TABLE "public"."patient_info"');
        await queryRunner.query('DROP TYPE "public"."patient_info_gender_enum"');
        await queryRunner.query('DROP INDEX "public"."IDX_a8398b11b6e9516e92c8c96b92"');
        await queryRunner.query('DROP INDEX "public"."IDX_4889b4fb25a2db3cd74ddc51ad"');
        await queryRunner.query('DROP TABLE "public"."diagnostic_report"');
        await queryRunner.query('DROP TABLE "public"."cough_characteristics"');
        await queryRunner.query('DROP TABLE "public"."cough_transitivity_types"');
        await queryRunner.query('DROP TYPE "public"."cough_transitivity_types_transitivity_type_enum"');
        await queryRunner.query('DROP TABLE "public"."cough_productivity_types"');
        await queryRunner.query('DROP TYPE "public"."cough_productivity_types_productivity_type_enum"');
        await queryRunner.query('DROP TABLE "public"."cough_intensity_types"');
        await queryRunner.query('DROP TYPE "public"."cough_intensity_types_intensity_type_enum"');
        await queryRunner.query('DROP TABLE "public"."cough_audio"');
        await queryRunner.query('DROP INDEX "public"."IDX_a9e0f73503ee7274446131da1b"');
        await queryRunner.query('DROP INDEX "public"."IDX_8239e9f19c1aedde804a20bd0b"');
        await queryRunner.query('DROP TABLE "public"."diagnostic_requests"');
        await queryRunner.query('DROP TABLE "public"."diagnostic_request_status"');
        await queryRunner.query('DROP TYPE "public"."diagnostic_request_status_request_status_enum"');
    }
}
