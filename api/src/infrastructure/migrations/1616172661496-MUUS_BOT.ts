import {MigrationInterface, QueryRunner} from "typeorm";

export class MUUSBOT1616172661496 implements MigrationInterface {
    name = 'MUUSBOT1616172661496'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`CREATE TABLE "public"."muus_diagnostic_request" ("id" SERIAL NOT NULL, "request_id" integer, "chat_id" integer NOT NULL, "status_id" integer NOT NULL, "dateCreated" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP, "date_finished" TIMESTAMP WITH TIME ZONE, "age" integer, "gender_id" integer, "is_smoking" boolean, "cough_audio_path" character varying, "is_forced" boolean, CONSTRAINT "REL_a2840bfdd94d2aec4e800ed6bd" UNIQUE ("request_id"), CONSTRAINT "PK_3a69cc255cb252746ad80faaa42" PRIMARY KEY ("id"))`);
        await queryRunner.query(`CREATE INDEX "IDX_a2840bfdd94d2aec4e800ed6bd" ON "public"."muus_diagnostic_request" ("request_id") `);
        await queryRunner.query(`CREATE INDEX "IDX_c8de56e2cb6907255e8af4a829" ON "public"."muus_diagnostic_request" ("chat_id") `);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" ADD CONSTRAINT "FK_a2840bfdd94d2aec4e800ed6bd5" FOREIGN KEY ("request_id") REFERENCES "public"."diagnostic_requests"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" ADD CONSTRAINT "FK_c8de56e2cb6907255e8af4a829d" FOREIGN KEY ("chat_id") REFERENCES "public"."bot_users"("chat_id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" ADD CONSTRAINT "FK_7ad71753ff2e1e7b69a161ec1e5" FOREIGN KEY ("status_id") REFERENCES "public"."tg_diagnostic_requests_status"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" ADD CONSTRAINT "FK_e76072439e8591b25eb52485ac6" FOREIGN KEY ("gender_id") REFERENCES "public"."gender_types"("id") ON DELETE NO ACTION ON UPDATE NO ACTION`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" DROP CONSTRAINT "FK_e76072439e8591b25eb52485ac6"`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" DROP CONSTRAINT "FK_7ad71753ff2e1e7b69a161ec1e5"`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" DROP CONSTRAINT "FK_c8de56e2cb6907255e8af4a829d"`);
        await queryRunner.query(`ALTER TABLE "public"."muus_diagnostic_request" DROP CONSTRAINT "FK_a2840bfdd94d2aec4e800ed6bd5"`);
        await queryRunner.query(`DROP INDEX "public"."IDX_c8de56e2cb6907255e8af4a829"`);
        await queryRunner.query(`DROP INDEX "public"."IDX_a2840bfdd94d2aec4e800ed6bd"`);
        await queryRunner.query(`DROP TABLE "public"."muus_diagnostic_request"`);
    }

}
