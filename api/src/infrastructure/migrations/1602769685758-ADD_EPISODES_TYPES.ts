import {MigrationInterface, QueryRunner} from 'typeorm';

export class ADDEPISODESTYPES1602769685758 implements MigrationInterface {
    name = 'ADDEPISODESTYPES1602769685758'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TYPE "public"."dataset_episodes_types_episode_type_enum" AS ENUM(\'breathing_inhale\', \'breathing_exhale\', \'other\')');
        await queryRunner.query('CREATE TABLE "public"."dataset_episodes_types" ("id" SERIAL NOT NULL, "episode_type" "public"."dataset_episodes_types_episode_type_enum" NOT NULL, CONSTRAINT "UQ_d877c61556a474978c919339137" UNIQUE ("episode_type"), CONSTRAINT "PK_d981265ee6b8e0dd077080ed39b" PRIMARY KEY ("id"))');
        await queryRunner.query('insert into "public"."dataset_episodes_types" ("id", "episode_type") values (1, \'breathing_inhale\'), (2, \'breathing_exhale\'), (3, \'other\')');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_espisodes" ADD "episode_type_id" integer NOT NULL DEFAULT 3');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_espisodes" ADD CONSTRAINT "FK_fd58e64ae3739519fa485bcc780" FOREIGN KEY ("episode_type_id") REFERENCES "public"."dataset_episodes_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_espisodes" DROP CONSTRAINT "FK_fd58e64ae3739519fa485bcc780"');
        await queryRunner.query('ALTER TABLE "public"."dataset_audio_espisodes" DROP COLUMN "episode_type_id"');
        await queryRunner.query('DROP TABLE "public"."dataset_episodes_types"');
        await queryRunner.query('DROP TYPE "public"."dataset_episodes_types_episode_type_enum"');
    }

}
