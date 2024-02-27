import {Router} from 'express';

import {createCoughDetectionCheck, createHWDiagnostic, createNoisyCoughCheck, 
    createCoughValidationCheck, createVoiceEmbedding, createVoiceComparisonCheck} from '../../controllers/hardwareController';
import * as bodyParser from 'body-parser';
import fileUpload from 'express-fileupload';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';

const router = Router();
const fileUploaderMiddleware = fileUpload();
const jsonBodyParser = bodyParser.json();
const ROUTE_NAME = 'diagnostic';
const HW_ROUTE = 'hardware';
const BASIC_MIDDLEWARES = [verifyTokenMiddleware, fileUploaderMiddleware];

router.post(`/${HW_ROUTE}/${ROUTE_NAME}`, BASIC_MIDDLEWARES, createHWDiagnostic);
router.post(`/${HW_ROUTE}/detector/cough`, BASIC_MIDDLEWARES, createCoughDetectionCheck);
router.post(`/${HW_ROUTE}/noisy/cough`, BASIC_MIDDLEWARES, createNoisyCoughCheck);
router.post(`/${HW_ROUTE}/validation/cough`, BASIC_MIDDLEWARES, createCoughValidationCheck);
router.post(`/${HW_ROUTE}/voice/embedding`, BASIC_MIDDLEWARES, createVoiceEmbedding);
router.post(`/${HW_ROUTE}/voice/compare`, BASIC_MIDDLEWARES, createVoiceComparisonCheck);

export default router;
