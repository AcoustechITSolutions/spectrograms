import {Router} from 'express';
import auth from './authRoute';
import diagnostic from './diagnosticRoute';
import dataset from './datasetRoute';
import diseaseTypes from './diseaseRoutes';
import markingRoute from './markingRoute';
import processingRoute from './processingRoute';
import hwRoute from './hwRoute';
import diagnosticBotApiRoute from './diagnosticBotApiRoute';

const router = Router();
const API_VERSION = 'v1';

router.use(`/${API_VERSION}`, auth);
router.use(`/${API_VERSION}`, diagnostic);
router.use(`/${API_VERSION}`, dataset);
router.use(`/${API_VERSION}`, diseaseTypes);
router.use(`/${API_VERSION}`, markingRoute);
router.use(`/${API_VERSION}`, processingRoute);
router.use(`/${API_VERSION}`, hwRoute);
router.use(`/${API_VERSION}`, diagnosticBotApiRoute);

export default router;