import {MigrationInterface, QueryRunner} from "typeorm";

export class NOISETYPES1625762486025 implements MigrationInterface {
    name = 'NOISETYPES1625762486025'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TYPE "public"."noise_types_noise_type_enum" AS ENUM('no_noise', 'little_noise', 'much_noise')`);
        await queryRunner.query(`CREATE TABLE "public"."noise_types" ("id" SERIAL NOT NULL, "noise_type" "public"."noise_types_noise_type_enum" NOT NULL, CONSTRAINT "UQ_9d446bbe267972726c1784ba9d7" UNIQUE ("noise_type"), CONSTRAINT "PK_fdb227a60eb9bc35a9139fc7d11" PRIMARY KEY ("id"))`);
        await queryRunner.query(`ALTER TABLE "public"."dataset_audio_info" ADD "noise_type_id" integer`);
        await queryRunner.query(`ALTER TABLE "public"."dataset_audio_info" ADD CONSTRAINT "FK_8b3b1fc933cbbaead42544422d0" FOREIGN KEY ("noise_type_id") REFERENCES "public"."noise_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."dataset_audio_info" DROP CONSTRAINT "FK_8b3b1fc933cbbaead42544422d0"`);
        await queryRunner.query(`ALTER TABLE "public"."dataset_audio_info" DROP COLUMN "noise_type_id"`);
        await queryRunner.query(`DROP TABLE "public"."noise_types"`);
        await queryRunner.query(`DROP TYPE "public"."noise_types_noise_type_enum"`);
    }

}
