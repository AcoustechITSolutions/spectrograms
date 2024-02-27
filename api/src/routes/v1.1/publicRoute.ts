import {Router} from 'express';
import {inferenceController} from '../../controllers/inferenceController';
import {createCoughDetectionCheck, createNoisyCoughCheck, createCoughValidationCheck} from '../../controllers/hardwareController';
import * as bodyParser from 'body-parser';
import fileUpload from 'express-fileupload';

const router = Router();
const fileUploaderMiddleware = fileUpload();
const jsonBodyParser = bodyParser.json();
const ROUTE_NAME = 'inference';

router.post(`/public/${ROUTE_NAME}`, [fileUploaderMiddleware], inferenceController);
router.post(`/public/detector/cough`, [fileUploaderMiddleware], createCoughDetectionCheck);
router.post(`/public/noisy/cough`, [fileUploaderMiddleware], createNoisyCoughCheck);
router.post(`/public/validation/cough`, [fileUploaderMiddleware], createCoughValidationCheck);

export default router;
