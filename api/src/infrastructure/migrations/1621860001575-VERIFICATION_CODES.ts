import {MigrationInterface, QueryRunner} from "typeorm";

export class VERIFICATIONCODES1621860001575 implements MigrationInterface {
    name = 'VERIFICATIONCODES1621860001575'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TABLE "public"."verification_codes" ("id" SERIAL NOT NULL, "user_id" integer NOT NULL, "code" integer NOT NULL, "date_created" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, CONSTRAINT "PK_d4bcbac7bacfeb5abc4b5982934" PRIMARY KEY ("id"))`);
        await queryRunner.query(`CREATE INDEX "IDX_fa2cd5944a73dd96d9ec50785f" ON "public"."verification_codes" ("user_id") `);
        await queryRunner.query(`ALTER TABLE "public"."users" ALTER COLUMN "email" DROP NOT NULL`);
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" ADD CONSTRAINT "FK_fa2cd5944a73dd96d9ec50785fa" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."verification_codes" DROP CONSTRAINT "FK_fa2cd5944a73dd96d9ec50785fa"`);
        await queryRunner.query(`ALTER TABLE "public"."users" ALTER COLUMN "email" SET NOT NULL`);
        await queryRunner.query(`DROP INDEX "public"."IDX_fa2cd5944a73dd96d9ec50785f"`);
        await queryRunner.query(`DROP TABLE "public"."verification_codes"`);
    }

}
