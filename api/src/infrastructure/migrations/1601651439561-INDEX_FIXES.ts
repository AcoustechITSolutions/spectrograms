import {MigrationInterface, QueryRunner} from 'typeorm';

export class INDEXFIXES1601651439561 implements MigrationInterface {
    name = 'INDEXFIXES1601651439561'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('DROP INDEX "public"."IDX_a9e0f73503ee7274446131da1b"');
        await queryRunner.query('CREATE INDEX "IDX_f3a0c242a6c2df1acca0241e15" ON "public"."patient_info" ("request_id") ');
        await queryRunner.query('CREATE INDEX "IDX_946955130a75b3e243ba690bb8" ON "public"."dataset_cough_characteristics" ("request_id") ');
        await queryRunner.query('CREATE INDEX "IDX_8bc85a48895c9598ab02654150" ON "public"."hw_cough_audio" ("request_id") ');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('DROP INDEX "public"."IDX_8bc85a48895c9598ab02654150"');
        await queryRunner.query('DROP INDEX "public"."IDX_946955130a75b3e243ba690bb8"');
        await queryRunner.query('DROP INDEX "public"."IDX_f3a0c242a6c2df1acca0241e15"');
        await queryRunner.query('CREATE INDEX "IDX_a9e0f73503ee7274446131da1b" ON "public"."diagnostic_requests" ("status_id") ');
    }
}
