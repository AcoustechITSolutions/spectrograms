import {MigrationInterface, QueryRunner} from "typeorm";

export class NULLABLEDATA1632909534908 implements MigrationInterface {
    name = 'NULLABLEDATA1632909534908'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" DROP COLUMN "is_dataset"`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd"`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "age" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "gender_id" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "is_smoking" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd"`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "is_smoking" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "gender_id" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ALTER COLUMN "age" SET NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" ADD "is_dataset" boolean NOT NULL DEFAULT false`);
    }

}
