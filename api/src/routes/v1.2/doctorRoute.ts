import {Router} from 'express';

import fileUpload from 'express-fileupload';
import {DoctorDiagnosticController} from '../../controllers/doctorDiagnosticController';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';

const router = Router();
const controller = new DoctorDiagnosticController();
const fileUploaderMiddleware = fileUpload();
const RESOURCE_NAME = 'doctor';
const ROUTE_NAME = 'diagnostic';
const BASIC_MIDDLEWARES = [verifyTokenMiddleware, checkRole([UserRoleTypes.EXTERNAL_SERVER])];

router.post(`/${RESOURCE_NAME}/${ROUTE_NAME}`, 
    [...BASIC_MIDDLEWARES, fileUploaderMiddleware], controller.createDiagnostic.bind(controller));
router.get(`/${RESOURCE_NAME}/${ROUTE_NAME}`, 
    [...BASIC_MIDDLEWARES, parsePaginationParams], controller.getDiagnostic.bind(controller));
router.get(`/${RESOURCE_NAME}/${ROUTE_NAME}/:id/`, 
    BASIC_MIDDLEWARES, controller.getDiagnosticById.bind(controller));
router.get(`/${RESOURCE_NAME}/${ROUTE_NAME}/:id/cough_audio`, 
    BASIC_MIDDLEWARES, controller.getDiagnosticCoughAudio.bind(controller));
router.put(`/${RESOURCE_NAME}/${ROUTE_NAME}/:id/cough_audio/`, [...BASIC_MIDDLEWARES, fileUploaderMiddleware], 
    controller.updateCoughAudio.bind(controller)
);

export default router;
