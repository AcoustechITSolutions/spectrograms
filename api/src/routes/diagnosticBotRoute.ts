import {Router} from 'express';
import {onDiagnosticBotMessage} from '../controllers/diagnosticBotController';
import {token} from '../container';

import * as bodyParser from 'body-parser';

const router = Router();
const jsonParser = bodyParser.json();

router.post(`/bot${token}`, [jsonParser], onDiagnosticBotMessage);

export default router;
