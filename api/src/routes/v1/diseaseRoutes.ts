import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {getAcuteCoughTypes, getChronicCoughTypes, getDiseaseTypes} from '../../controllers/diseaseTypesController';

const router = Router();
const ROUTE_NAME = 'disease_types';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated];

router.get(`/${ROUTE_NAME}`, BASIC_MIDDLEWARES, getDiseaseTypes);
router.get(`/${ROUTE_NAME}/acute`, BASIC_MIDDLEWARES, getAcuteCoughTypes);
router.get(`/${ROUTE_NAME}/chronic`, BASIC_MIDDLEWARES, getChronicCoughTypes);

export default router;
