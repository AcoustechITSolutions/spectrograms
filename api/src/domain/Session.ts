import {sign, verify, decode} from 'jsonwebtoken';
import moment from 'moment';
import {unitOfTime} from 'moment';
import {getConnection} from 'typeorm';
import config from '../config/config';
import {RefreshToken} from '../infrastructure/entity/RefreshToken';
import {User} from '../infrastructure/entity/Users';
import {UserRoleTypes} from '../domain/UserRoles';

export interface Session {
    id: number;
    dateCreated: number;
    username: string;
    issued: number;
    expires: number;
}

export interface DecodedToken {
    userId: number,
    roles: UserRoleTypes[],
    jwtId: string,
    jti: string,
    iat: number,
    exp: number
}

export type Token = Omit<DecodedToken, 'iat'>

export type PartialSession = Omit<Session, 'issued' | 'expires'>;

export interface EncodeResult {
    token: string,
    expires: number,
    issued: number
}

export type DecodeResult =
    | {
          type: 'valid';
          session: Session;
      }
    | {
          type: 'integrity-error';
      }
    | {
          type: 'invalid-token';
      };

export type ExpirationStatus = 'expired' | 'active' | 'grace';

export const generateSignToken = (token: Token) => {
    return sign(token, config.jwtSecret, {
        expiresIn: config.jwtLife,
        algorithm: 'HS256',
    });
};

export const verifyToken = (
    token: any,
    ignoreExpiration: boolean = false,
    ignoreNotBefore: boolean = false,
): DecodedToken | undefined => {
    try {
        const decodedToken = verify(token, config.jwtSecret, {
            algorithms: ['HS256'],
            ignoreExpiration,
            ignoreNotBefore,
        }) as DecodedToken;
        return decodedToken;
    } catch (error) {
        return undefined;
    }
};

export const decodeToken = (token: string): DecodedToken | undefined => {
    try {
        return decode(token) as DecodedToken;
    } catch (error) {
        console.error(error);
        return undefined;
    }
};

export const generateRefreshToken = async (user: User, jwtId: string): Promise<RefreshToken> => {
    const refreshToken = new RefreshToken();
    refreshToken.user = user;
    refreshToken.jwt_id = jwtId;
    const period = config.jwtRefreshLife.match(/[^\d.-]/g).join() as unitOfTime.Base;
    refreshToken.expires_date = moment().add(
        parseInt(config.jwtRefreshLife),
        period
    ).toDate();
    
    const connection = getConnection();    
    const refreshTokenRepo = connection.getRepository(RefreshToken);
    await refreshTokenRepo.save(refreshToken);
    return refreshToken;
};

export const verifyRefreshToken = async (
    refreshTokenId: string,
    token: Token
): Promise<RefreshToken> | undefined => {
    const connection = getConnection();    
    const refreshTokenRepo = connection.getRepository(RefreshToken);
    let refreshToken: RefreshToken;
    try {
        refreshToken = await refreshTokenRepo.findOneOrFail(refreshTokenId);
        const isJwtIdNotMatched: boolean = refreshToken.jwt_id !== token.jwtId;
        const isJwtIdExpired: boolean = moment().isAfter(refreshToken.expires_date);
        if (isJwtIdNotMatched || isJwtIdExpired) {
            return undefined;
        }
        return refreshToken;
    } catch (error) {
        return undefined;
    }
};
