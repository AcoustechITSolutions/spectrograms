import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {checkRole} from '../../middlewares/checkRole';
import * as bodyParser from 'body-parser';
import {UserRoleTypes} from '../../domain/UserRoles';
import {ProcessingController} from '../../controllers/processingController';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';

const router = Router();
const controller = new ProcessingController();
const jsonBodyParser = bodyParser.json();
const ROUTE_NAME = 'processing';
const ADMIN_ROUTE = 'admin';

router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER]), parsePaginationParams], 
    controller.getProcessing.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER])], 
    controller.getProcessingById.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/cough_audio/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER])], 
    controller.getProcessingCoughAudio.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/patient_photo/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER])], 
    controller.getProcessingPatientPhoto.bind(controller));
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, [verifyTokenMiddlewareDeprecated, jsonBodyParser, checkRole([UserRoleTypes.EDIFIER])],
    controller.patchProcessing.bind(controller));
router.delete(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER])],
    controller.deleteProcessing.bind(controller));
router.post(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/noisy/`, [verifyTokenMiddlewareDeprecated, jsonBodyParser, checkRole([UserRoleTypes.EDIFIER])], 
    controller.markRecordNoisy.bind(controller));

export default router;
