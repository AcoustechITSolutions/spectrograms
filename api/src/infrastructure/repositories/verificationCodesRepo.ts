import {EntityRepository, Repository, getConnection} from 'typeorm';
import {VerificationCodes} from '../entity/VerificationCodes';

@EntityRepository(VerificationCodes)
export class VerificationCodesRepository extends Repository<VerificationCodes> {
    public async findCodeByUserId(userId: number): Promise<VerificationCodes> {
        const connection = getConnection();
        const codes = await connection.createQueryBuilder(VerificationCodes, 'codes')
            .select('*')
            .where('codes.user_id = :myUserId', {myUserId: userId})
            .orderBy('codes.date_expired', 'DESC')
            .execute();

        return codes[0];
    }
}