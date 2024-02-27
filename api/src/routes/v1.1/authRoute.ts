import {Router} from 'express';
import {AuthController} from '../../controllers/authController';
import * as bodyParser from 'body-parser';
import {verifyTokenMiddleware} from '../../middlewares/verifyAuth';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';

const controller = new AuthController();
const router = Router();
const jsonParser = bodyParser.json();
const ROUTE_NAME = 'login';
const ADMIN_ROUTE = 'admin';
const TOKEN_ROUTE = 'token';
const USER_ROUTE = 'user';
const BASIC_MIDDLEWARES = [verifyTokenMiddleware, checkRole([UserRoleTypes.ADMIN])];

router.post(`/${ROUTE_NAME}`, [jsonParser], controller.login.bind(controller));
router.post(`/${ADMIN_ROUTE}/${ROUTE_NAME}`, [jsonParser], controller.adminLogin.bind(controller));
router.post(`/${ADMIN_ROUTE}/${USER_ROUTE}`, [jsonParser], BASIC_MIDDLEWARES, controller.registerUser.bind(controller));
router.patch(`/${ADMIN_ROUTE}/${USER_ROUTE}/:id`, [jsonParser], BASIC_MIDDLEWARES, controller.patchUser.bind(controller));
router.post(`/${TOKEN_ROUTE}/refresh`, [jsonParser], controller.refreshToken.bind(controller));
router.post(`/${TOKEN_ROUTE}/logout`, [jsonParser], controller.logout.bind(controller));
router.post(`/${USER_ROUTE}`, [jsonParser], controller.selfRegister.bind(controller));
router.post(`/${USER_ROUTE}/send_code`, [jsonParser], controller.sendCode.bind(controller));
router.post(`/${USER_ROUTE}/:id/verify`, [jsonParser], controller.verifyCode.bind(controller));
router.post(`/${USER_ROUTE}/:id/password`, [jsonParser], controller.setPassword.bind(controller));
router.patch(`/${USER_ROUTE}/password`, [jsonParser], verifyTokenMiddleware, controller.changePassword.bind(controller));
router.get(`/${USER_ROUTE}/data`, verifyTokenMiddleware, controller.getPersonalData.bind(controller));
router.patch(`/${USER_ROUTE}/data`, [jsonParser], verifyTokenMiddleware, controller.patchPersonalData.bind(controller));

export default router;
