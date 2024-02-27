import {Router} from 'express';
import {onDiagnosticBotMessage} from '../controllers/muusBotController';
import {muusToken} from '../container';

import * as bodyParser from 'body-parser';

const router = Router();
const jsonParser = bodyParser.json();

router.post(`/bot${muusToken}`, [jsonParser], onDiagnosticBotMessage);

export default router;
