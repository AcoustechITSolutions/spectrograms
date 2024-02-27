import {MigrationInterface, QueryRunner} from "typeorm";

export class DOCTORSPATIENTS1623344976353 implements MigrationInterface {
    name = 'DOCTORSPATIENTS1623344976353'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TABLE "public"."doctors_patients" ("doctor_id" integer NOT NULL, "patient_id" integer NOT NULL, CONSTRAINT "PK_2436fee387c5e4ab33467f95cc1" PRIMARY KEY ("doctor_id", "patient_id"))`);
        await queryRunner.query(`CREATE UNIQUE INDEX "IDX_2436fee387c5e4ab33467f95cc" ON "public"."doctors_patients" ("doctor_id", "patient_id") `);
        await queryRunner.query(`ALTER TABLE "public"."doctors_patients" ADD CONSTRAINT "FK_04e5da63a9e116b8de5f8bca94f" FOREIGN KEY ("doctor_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."doctors_patients" ADD CONSTRAINT "FK_a050cc7ff85c483c56a06c82025" FOREIGN KEY ("patient_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."doctors_patients" DROP CONSTRAINT "FK_a050cc7ff85c483c56a06c82025"`);
        await queryRunner.query(`ALTER TABLE "public"."doctors_patients" DROP CONSTRAINT "FK_04e5da63a9e116b8de5f8bca94f"`);
        await queryRunner.query(`DROP INDEX "public"."IDX_2436fee387c5e4ab33467f95cc"`);
        await queryRunner.query(`DROP TABLE "public"."doctors_patients"`);
    }

}
