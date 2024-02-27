import {Router} from 'express';
import {onDiagnosticBotMessage} from '../controllers/newDiagnosticBotController';
import {diagnosticToken} from '../container';

import * as bodyParser from 'body-parser';

const router = Router();
const jsonParser = bodyParser.json();

router.post(`/bot${diagnosticToken}`, [jsonParser], onDiagnosticBotMessage);

export default router;
