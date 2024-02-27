import {Request, Response, NextFunction} from 'express';
import {getRepository} from 'typeorm';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {User} from '../infrastructure/entity/Users';
import {UserRoleTypes, isEitherRolesMatch, isRolesMatch} from '../domain/UserRoles';

// Statisfies even if not all roles belongs to user
export const checkEitherRole = (roles: Array<UserRoleTypes>) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        const userRoles = await getUserRoles(req.token.userId);
        if (userRoles == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
        const isMatch = isEitherRolesMatch(roles, userRoles);
        if (isMatch) {
            next();
        } else {
            const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
            return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
        }
    };
};

export const checkRole = (roles: Array<UserRoleTypes>) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        const userRoles = await getUserRoles(req.token.userId);
        if (userRoles == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const isMatch = isRolesMatch(roles, userRoles);
        if (isMatch) {
            next();
        } else {
            const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
            return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
        }
    };
};

export const blockForRoles = (roles: Array<UserRoleTypes>) => {
    return async (req: Request, res: Response, next: NextFunction) => {
        const userRoles = await getUserRoles(req.token.userId);
        if (userRoles == null) {
            const errorMessage = getErrorMessage(HttpErrors.NO_USER);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const isMatch = isEitherRolesMatch(roles, userRoles);
        if (isMatch) {
            const errorMessage = getErrorMessage(HttpErrors.FORBIDDEN);
            return res.status(HttpStatusCodes.FORBIDDEN).send(errorMessage);
        } else {
            next();
        }
    };
};

export const getUserRoles = async (userId: number): Promise<UserRoleTypes[]> | null => {
    const rolesRepository = getRepository(User);

    try {
        const user = await rolesRepository.findOneOrFail(userId, {relations: ['roles']});
        return user
            .roles
            .map((row) => row.role);
    } catch (error) {
        console.error(error);
        return null;
    }
};
