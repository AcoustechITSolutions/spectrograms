import {MigrationInterface, QueryRunner} from "typeorm";

export class LOCATIONCOORDINATES1631869760429 implements MigrationInterface {
    name = 'LOCATIONCOORDINATES1631869760429'

    public async up(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" ADD "location_latitude" double precision`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" ADD "location_longitude" double precision`);
    }

    public async down(queryRunner: QueryRunner): Promise<void> {
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" DROP COLUMN "location_longitude"`);
        await queryRunner.query(`ALTER TABLE "public"."diagnostic_requests" DROP COLUMN "location_latitude"`);
    }

}
