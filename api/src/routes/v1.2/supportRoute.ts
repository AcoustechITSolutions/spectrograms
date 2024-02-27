import {Router} from 'express';
import {SupportController} from '../../controllers/supportController';
import * as bodyParser from 'body-parser';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';

const controller = new SupportController();
const router = Router();
const jsonParser = bodyParser.json();
const ROUTE_NAME = 'support';

router.post(`/${ROUTE_NAME}/unauthorized`, [jsonParser], controller.requestSupportUnauthorized.bind(controller));
router.post(`/${ROUTE_NAME}`, [jsonParser], verifyTokenMiddleware, controller.requestSupportAuthorized.bind(controller));
router.post(`/${ROUTE_NAME}/discharge`, verifyTokenMiddleware, controller.requestDataDischarge.bind(controller));

export default router;