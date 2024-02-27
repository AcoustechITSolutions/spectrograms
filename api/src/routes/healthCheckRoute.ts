import express from 'express';
import {HttpStatusCodes} from '../helpers/status';

const router = express.Router();

router.get('/health', (req, res) => {
    res.status(HttpStatusCodes.SUCCESS).send('Working');
});

export default router;
