import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXDIAGNOSTICREPORT1596463065678 implements MigrationInterface {
    name = 'FIXDIAGNOSTICREPORT1596463065678'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP CONSTRAINT "FK_4889b4fb25a2db3cd74ddc51ad0"');
        await queryRunner.query('DROP INDEX "public"."IDX_4889b4fb25a2db3cd74ddc51ad"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "requestId" TO "request_id"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME CONSTRAINT "REL_4889b4fb25a2db3cd74ddc51ad" TO "UQ_9542ddbab16b4a838798f674e47"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "request_id" SET NOT NULL');
        await queryRunner.query('CREATE INDEX "IDX_9542ddbab16b4a838798f674e4" ON "public"."diagnostic_report" ("request_id") ');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD CONSTRAINT "FK_9542ddbab16b4a838798f674e47" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" DROP CONSTRAINT "FK_9542ddbab16b4a838798f674e47"');
        await queryRunner.query('DROP INDEX "public"."IDX_9542ddbab16b4a838798f674e4"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ALTER COLUMN "request_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME CONSTRAINT "UQ_9542ddbab16b4a838798f674e47" TO "REL_4889b4fb25a2db3cd74ddc51ad"');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" RENAME COLUMN "request_id" TO "requestId"');
        await queryRunner.query('CREATE INDEX "IDX_4889b4fb25a2db3cd74ddc51ad" ON "public"."diagnostic_report" ("requestId") ');
        await queryRunner.query('ALTER TABLE "public"."diagnostic_report" ADD CONSTRAINT "FK_4889b4fb25a2db3cd74ddc51ad0" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
