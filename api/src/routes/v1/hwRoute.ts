import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {createCoughDetectionCheck, createHWDiagnostic, createNoisyCoughCheck, createCoughValidationCheck} from '../../controllers/hardwareController';
import * as bodyParser from 'body-parser';
import fileUpload from 'express-fileupload';

const router = Router();
const fileUploaderMiddleware = fileUpload();
const jsonBodyParser = bodyParser.json();
const ROUTE_NAME = 'diagnostic';
const HW_ROUTE = 'hardware';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, fileUploaderMiddleware];

router.post(`/${HW_ROUTE}/${ROUTE_NAME}`, BASIC_MIDDLEWARES, createHWDiagnostic);
router.post(`/${HW_ROUTE}/detector/cough`, BASIC_MIDDLEWARES, createCoughDetectionCheck);
router.post(`/${HW_ROUTE}/noisy/cough`, BASIC_MIDDLEWARES, createNoisyCoughCheck);
router.post(`/${HW_ROUTE}/validation/cough`, BASIC_MIDDLEWARES, createCoughValidationCheck);

export default router;
