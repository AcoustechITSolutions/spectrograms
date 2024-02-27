import {MigrationInterface, QueryRunner} from 'typeorm';

export class AUTOGENERATE1613074823568 implements MigrationInterface {
    name = 'AUTOGENERATE1613074823568'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('CREATE TABLE "public"."notify_diagnostic_bot_users" ("id" SERIAL NOT NULL, "chat_id" integer NOT NULL, "language" character varying NOT NULL DEFAULT \'ru\', CONSTRAINT "UQ_2528ef693859db3bfe9761b60a8" UNIQUE ("chat_id"), CONSTRAINT "PK_209b1cf7e1ccbe777507c7f1cff" PRIMARY KEY ("id"))');
        await queryRunner.query('CREATE INDEX "IDX_2528ef693859db3bfe9761b60a" ON "public"."notify_diagnostic_bot_users" ("chat_id") ');
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query('DROP INDEX "public"."IDX_2528ef693859db3bfe9761b60a"');
        await queryRunner.query('DROP TABLE "public"."notify_diagnostic_bot_users"');
    }

}
