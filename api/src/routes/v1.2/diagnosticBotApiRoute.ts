import {Router} from 'express';
import fileUpload from 'express-fileupload';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';
import {createDiagnostic, getDiagnosticResultById, getDiagnosticResults} from '../../controllers/diagnosticBotApiController';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';

const router = Router();
const fileUploaderMiddleware = fileUpload();
const TG_BOT_ROUTE = 'diagnostic_bot';
const BASIC_MIDDLEWARES = [verifyTokenMiddleware, checkRole([UserRoleTypes.PATIENT])];

router.post(`/${TG_BOT_ROUTE}/diagnostic`, [...BASIC_MIDDLEWARES, fileUploaderMiddleware], createDiagnostic);
router.get(`/${TG_BOT_ROUTE}/diagnostic/:id/`, [...BASIC_MIDDLEWARES], getDiagnosticResultById);
router.get(`/${TG_BOT_ROUTE}/diagnostic`, [...BASIC_MIDDLEWARES, parsePaginationParams], getDiagnosticResults);

export default router;
