import {MigrationInterface, QueryRunner} from "typeorm";

export class PERSONALDATA1624532743513 implements MigrationInterface {
    name = 'PERSONALDATA1624532743513'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TABLE "public"."personal_data" ("id" SERIAL NOT NULL, "user_id" integer NOT NULL, "identifier" character varying, "age" integer NOT NULL, "gender_id" integer NOT NULL, "is_smoking" boolean NOT NULL, CONSTRAINT "REL_fa62725914b20b9f120e0ffaba" UNIQUE ("user_id"), CONSTRAINT "PK_64de2c49d93d7ff4d4fe3c61873" PRIMARY KEY ("id"))`);
        await queryRunner.query(`CREATE INDEX "IDX_fa62725914b20b9f120e0ffaba" ON "public"."personal_data" ("user_id") `);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD CONSTRAINT "FK_fa62725914b20b9f120e0ffabae" FOREIGN KEY ("user_id") REFERENCES "public"."users"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" ADD CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP CONSTRAINT "FK_38222bfbcae82ae2a91692a0efd"`);
        await queryRunner.query(`ALTER TABLE "public"."personal_data" DROP CONSTRAINT "FK_fa62725914b20b9f120e0ffabae"`);
        await queryRunner.query(`DROP INDEX "public"."IDX_fa62725914b20b9f120e0ffaba"`);
        await queryRunner.query(`DROP TABLE "public"."personal_data"`);
    }

}
