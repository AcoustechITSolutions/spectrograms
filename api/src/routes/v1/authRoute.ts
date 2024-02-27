import {Router} from 'express';
import {AuthController} from '../../controllers/authController';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';

import * as bodyParser from 'body-parser';

const controller = new AuthController();
const router = Router();
const jsonParser = bodyParser.json();
const ROUTE_NAME = 'login';
const ADMIN_ROUTE = 'admin';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.ADMIN])];

router.post(`/${ROUTE_NAME}`, [jsonParser], controller.login.bind(controller));
router.post(`/${ADMIN_ROUTE}/${ROUTE_NAME}`, [jsonParser], controller.adminLogin.bind(controller));
router.post(`/${ADMIN_ROUTE}/user`, [jsonParser], BASIC_MIDDLEWARES, controller.registerUser.bind(controller));
router.patch(`/${ADMIN_ROUTE}/user/:login`, [jsonParser], BASIC_MIDDLEWARES, controller.patchUser.bind(controller));
export default router;
