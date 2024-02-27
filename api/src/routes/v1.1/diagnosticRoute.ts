import {Router} from 'express';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';
import {DiagnosticController} from '../../controllers/diagnosticController';
import fileUpload from 'express-fileupload';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';

const controller = new DiagnosticController();
const router = Router();
const fileUploaderMiddleware = fileUpload();
const ROUTE_NAME = 'diagnostic';
const BASIC_MIDDLEWARES = [verifyTokenMiddleware, checkRole([UserRoleTypes.PATIENT])];

router.post(`/${ROUTE_NAME}`, [...BASIC_MIDDLEWARES, fileUploaderMiddleware], 
    controller.diagnosticNew.bind(controller));
router.get(`/${ROUTE_NAME}/`, [...BASIC_MIDDLEWARES, parsePaginationParams], 
    controller.getDiagnostic.bind(controller));
router.get(`/${ROUTE_NAME}/:id/`, BASIC_MIDDLEWARES, 
    controller.getDiagnosticById.bind(controller));
router.delete(`/${ROUTE_NAME}/:id/`, BASIC_MIDDLEWARES, 
    controller.deleteUserPdf.bind(controller));
router.put(`/${ROUTE_NAME}/:id/cough_audio/`, [...BASIC_MIDDLEWARES, fileUploaderMiddleware], 
    controller.updateCoughAudio.bind(controller));
router.get(`/${ROUTE_NAME}/report/:id/`, controller.getReportByQR.bind(controller))    

export default router;
