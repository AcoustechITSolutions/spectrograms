import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXCOUGHCHAR1596460470157 implements MigrationInterface {
    name = 'FIXCOUGHCHAR1596460470157'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dd304f9a097ae06778a2bab3e5b"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" RENAME COLUMN "requestId" TO "request_id"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" RENAME CONSTRAINT "REL_dd304f9a097ae06778a2bab3e5" TO "UQ_45b59d86f023130fb91ac2b033f"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "request_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_45b59d86f023130fb91ac2b033f" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_45b59d86f023130fb91ac2b033f"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "request_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" RENAME CONSTRAINT "UQ_45b59d86f023130fb91ac2b033f" TO "REL_dd304f9a097ae06778a2bab3e5"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" RENAME COLUMN "request_id" TO "requestId"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dd304f9a097ae06778a2bab3e5b" FOREIGN KEY ("requestId") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
