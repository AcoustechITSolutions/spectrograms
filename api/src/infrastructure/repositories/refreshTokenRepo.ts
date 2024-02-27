import {EntityRepository, Repository, getConnection} from 'typeorm';
import {RefreshToken} from '../entity/RefreshToken';
import moment from 'moment';

@EntityRepository(RefreshToken)
export class RefreshTokenRepository extends Repository<RefreshToken> {
    public async findByJwtIdOrFail(jwtId: string): Promise<RefreshToken> {
        return this.findOneOrFail({
            where: {jwt_id: jwtId}
        });
    }

    public async findActiveByUserId(userId: number): Promise<RefreshToken | undefined> {
        const connection = getConnection();
        const tokens = await connection.createQueryBuilder(RefreshToken, 'token')
            .select('*')
            .where('token.user_id = :myUserId', {myUserId: userId})
            .orderBy('token.expires_date', 'DESC')
            .execute();

        if (tokens.length == 0) 
            return undefined;
        if (moment().isAfter(tokens[0].expires_date)) 
            return undefined;

        return tokens[0];
    }
}
