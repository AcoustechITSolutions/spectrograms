import {Token} from '../../src/domain/Session'
import {PaginationParams} from '../../src/interfaces/PaginationParams'

declare global {
    namespace Express {
        interface Request {
            token: Token,
            paginationParams?: PaginationParams
        }
    }
}
