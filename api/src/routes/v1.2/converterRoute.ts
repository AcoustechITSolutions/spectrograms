import {Router} from 'express';

import fileUpload from 'express-fileupload';

import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';
import {ConverterController} from '../../controllers/converterController';

const router = Router();
const controller = new ConverterController();
const fileUploaderMiddleware = fileUpload();
const ROUTE_NAME = 'convert';

router.post(`/${ROUTE_NAME}`, [verifyTokenMiddleware, fileUploaderMiddleware], controller.convert.bind(controller));

export default router;