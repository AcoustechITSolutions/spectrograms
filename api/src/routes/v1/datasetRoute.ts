import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';

import fileUpload from 'express-fileupload';
import {DatasetController} from '../../controllers/datasetController';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';
import {fileService} from '../../container';
import {DatasetQueryService} from '../../services/query/DatasetQueryService';
import {checkRole} from '../../middlewares/checkRole';
import {UserRoleTypes} from '../../domain/UserRoles';

const queryService = new DatasetQueryService();
const controller = new DatasetController(fileService, queryService);
const router = Router();
const fileUploaderMiddleware = fileUpload();
const ROUTE_NAME = 'dataset';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.DATASET])];

router.post(`/${ROUTE_NAME}`, [...BASIC_MIDDLEWARES, fileUploaderMiddleware], controller.createDatasetNullable.bind(controller));
router.get(`/${ROUTE_NAME}`, [...BASIC_MIDDLEWARES, parsePaginationParams], controller.getDatasetRecords.bind(controller));
router.delete(`/${ROUTE_NAME}/:id`, BASIC_MIDDLEWARES, controller.deleteUserDataset.bind(controller));
export default router;
