import {Router} from 'express';
import {verifyTokenMiddlewareDeprecated} from '../../middlewares/verifyAuth';
import {checkRole, checkEitherRole} from '../../middlewares/checkRole';
import * as bodyParser from 'body-parser';
import {UserRoleTypes} from '../../domain/UserRoles';
import {getMarkingRecords, getAudioGeneralInfo, getCoughDetailedInfo,
    getBreathingDetailedInfo, getSpeechDetailedInfo, getAudio, getSpectrogram,
    patchGeneralInfo, patchStatusInfo, patchCoughCharacteristics,
    patchSpeechCharacteristics, patchBreathingCharacteristics,
    putAudioEpisodes, getNavigationById} from '../../controllers/markingController';
import {DatasetAudioTypes} from '../../domain/DatasetAudio';
import {parsePaginationParams} from '../../middlewares/parsePaginationParams';

const router = Router();
const jsonBodyParser = bodyParser.json();
const ADMIN_ROUTE = 'admin';
const ROUTE_NAME = 'marking';
const DETAILED_ROUTE = 'detailed';
const BASIC_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, checkEitherRole([UserRoleTypes.DATA_SCIENTIST, UserRoleTypes.DOCTOR])];
const SCIENTIST_MIDDLEWARES = [verifyTokenMiddlewareDeprecated, checkRole([UserRoleTypes.DATA_SCIENTIST])];

router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/`, [...BASIC_MIDDLEWARES, parsePaginationParams], getMarkingRecords);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/general/`, BASIC_MIDDLEWARES, getAudioGeneralInfo);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.COUGH}/${DETAILED_ROUTE}/`, BASIC_MIDDLEWARES, getCoughDetailedInfo);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.BREATHING}/${DETAILED_ROUTE}/`, BASIC_MIDDLEWARES, getBreathingDetailedInfo);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.SPEECH}/${DETAILED_ROUTE}/`, BASIC_MIDDLEWARES, getSpeechDetailedInfo);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/:type/audio/`, BASIC_MIDDLEWARES, getAudio);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/:type/spectrogram/`, BASIC_MIDDLEWARES, getSpectrogram);
router.get(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/navigation/`, BASIC_MIDDLEWARES, getNavigationById);

router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/general`, [...BASIC_MIDDLEWARES, jsonBodyParser], patchGeneralInfo);
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id`, [...BASIC_MIDDLEWARES, jsonBodyParser], patchStatusInfo);
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.COUGH}/${DETAILED_ROUTE}`, [...BASIC_MIDDLEWARES, jsonBodyParser], patchCoughCharacteristics);
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.SPEECH}/${DETAILED_ROUTE}`, [...BASIC_MIDDLEWARES, jsonBodyParser], patchSpeechCharacteristics);
router.patch(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/${DatasetAudioTypes.BREATHING}/${DETAILED_ROUTE}`, [...BASIC_MIDDLEWARES, jsonBodyParser], patchBreathingCharacteristics);

router.put(`/${ADMIN_ROUTE}/${ROUTE_NAME}/:id/:type/episodes`, [...SCIENTIST_MIDDLEWARES, jsonBodyParser], putAudioEpisodes);
export default router;
