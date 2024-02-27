import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDREQUESTIDTOPATIENT1596447019172 implements MigrationInterface {
    name = 'ADDREQUESTIDTOPATIENT1596447019172'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_81031c936ec7d07cbea5d347adc"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" RENAME COLUMN "requestId" TO "request_id"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" RENAME CONSTRAINT "REL_81031c936ec7d07cbea5d347ad" TO "UQ_f3a0c242a6c2df1acca0241e15d"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "request_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_f3a0c242a6c2df1acca0241e15d" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."patient_info" DROP CONSTRAINT "FK_f3a0c242a6c2df1acca0241e15d"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ALTER COLUMN "request_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."patient_info" RENAME CONSTRAINT "UQ_f3a0c242a6c2df1acca0241e15d" TO "REL_81031c936ec7d07cbea5d347ad"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" RENAME COLUMN "request_id" TO "requestId"');
        await queryRunner.query('ALTER TABLE "public"."patient_info" ADD CONSTRAINT "FK_81031c936ec7d07cbea5d347adc" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
