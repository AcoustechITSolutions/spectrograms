import {Request, Response, NextFunction} from 'express';
import {PaginationParams, DEFAULT_LIMIT, DEFAULT_OFFSET} from '../interfaces/PaginationParams';

export const parsePaginationParams = async (req: Request, res: Response, next: NextFunction) => {
    const paginationParams: PaginationParams = {
        offset: Number(req.query.offset ?? DEFAULT_OFFSET),
        limit: Number(req.query.limit ?? DEFAULT_LIMIT),
    };
    req.paginationParams = paginationParams;

    next();
};
