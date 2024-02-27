import {Router} from 'express';
import {onDatasetBotMessage} from '../controllers/datasetBotController';
import {dataToken} from '../container';

import * as bodyParser from 'body-parser';

const router = Router();
const jsonParser = bodyParser.json();

router.post(`/bot${dataToken}`, [jsonParser], onDatasetBotMessage);

export default router;
