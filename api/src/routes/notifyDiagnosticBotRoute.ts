import {Router} from 'express';
import {onNotifyDiagnosticBotMessage} from '../controllers/notifyDiagnosticBotController';
import {notifyDiagnosticToken} from '../container';

import * as bodyParser from 'body-parser';

const router = Router();
const jsonParser = bodyParser.json();

router.post(`/bot${notifyDiagnosticToken}`, [jsonParser], onNotifyDiagnosticBotMessage);

export default router;
