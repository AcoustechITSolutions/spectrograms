import {Router, Request, Response, NextFunction} from 'express';
import * as bodyParser from 'body-parser';
import {SnsController} from '../controllers/snsTopicsController';

const controller = new SnsController();
const router = Router();
const ROUTE_NAME = 'sns';
const jsonParser = bodyParser.json();
const manageHeaders = function(req: Request, res: Response, next: NextFunction) {
    if (req.get('x-amz-sns-message-type')) {
        req.headers['content-type'] = 'application/json'; 
    }
    next();
};

router.post(`/${ROUTE_NAME}/ses_complaints`, [manageHeaders, jsonParser], controller.handleSesFailures.bind(controller));
router.post(`/${ROUTE_NAME}/ses_bounces`, [manageHeaders, jsonParser], controller.handleSesFailures.bind(controller));
export default router;
