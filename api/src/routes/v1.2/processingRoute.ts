import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {checkRole, checkEitherRole} from '../../middlewares/checkRole';
import * as bodyParser from 'body-parser';
import {UserRoleTypes} from '../../domain/UserRoles';
import {ProcessingController} from '../../controllers/processingController';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';

const router = Router();
const controller = new ProcessingController();
const jsonBodyParser = bodyParser.json();
const ROUTE_NAME = 'processing';
const ADMIN_ROUTE = 'admin';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, checkEitherRole([UserRoleTypes.EDIFIER, UserRoleTypes.VIEWER])];

router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/`, [...BASIC_MIDDLEWARES, parsePaginationParams], 
    controller.getProcessing.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, BASIC_MIDDLEWARES, 
    controller.getProcessingById.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/statistics/diagnosis/`, BASIC_MIDDLEWARES, 
    controller.getProcessingStatistics.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/cough_audio/`, BASIC_MIDDLEWARES, 
    controller.getProcessingCoughAudio.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/patient_photo/`, BASIC_MIDDLEWARES, 
    controller.getProcessingPatientPhoto.bind(controller));
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/spectrogram/`, BASIC_MIDDLEWARES, 
    controller.getProcessingSpectrogram.bind(controller));
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, [verifyTokenMiddlewareDeprecated, jsonBodyParser, checkRole([UserRoleTypes.EDIFIER])],
    controller.patchProcessing.bind(controller));
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/details`, [verifyTokenMiddlewareDeprecated, jsonBodyParser, checkRole([UserRoleTypes.VIEWER])],
    controller.patchProcessingDetails.bind(controller));
router.delete(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/`, [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.EDIFIER])],
    controller.deleteProcessing.bind(controller));
router.post(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/noisy/`, [verifyTokenMiddlewareDeprecated, jsonBodyParser, checkRole([UserRoleTypes.EDIFIER])], 
    controller.markRecordNoisy.bind(controller));

export default router;
