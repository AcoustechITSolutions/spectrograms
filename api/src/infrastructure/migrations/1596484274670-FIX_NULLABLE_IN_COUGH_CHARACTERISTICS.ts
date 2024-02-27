import {MigrationInterface, QueryRunner} from 'typeorm';

export class FIXNULLABLEINCOUGHCHARACTERISTICS1596484274670 implements MigrationInterface {
    name = 'FIXNULLABLEINCOUGHCHARACTERISTICS1596484274670'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "productivity_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "transitivity_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "intensity_id" DROP NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541" FOREIGN KEY ("productivity_id") REFERENCES "public"."cough_productivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb" FOREIGN KEY ("transitivity_id") REFERENCES "public"."cough_transitivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26" FOREIGN KEY ("intensity_id") REFERENCES "public"."cough_intensity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" DROP CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541"');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "intensity_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "transitivity_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ALTER COLUMN "productivity_id" SET NOT NULL');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_28a990700c0e2ee1dc8f34d3e26" FOREIGN KEY ("intensity_id") REFERENCES "public"."cough_intensity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_dbe4442e2c6c555837bbb24ebdb" FOREIGN KEY ("transitivity_id") REFERENCES "public"."cough_transitivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
        await queryRunner.query('ALTER TABLE "public"."cough_characteristics" ADD CONSTRAINT "FK_ccb0c1f61d572cdcb5c20376541" FOREIGN KEY ("productivity_id") REFERENCES "public"."cough_productivity_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }
}
