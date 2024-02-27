import {Router} from 'express';
import {PaymentController} from '../controllers/paymentController';

const controller = new PaymentController();
const router = Router();
const ROUTE_NAME = 'payment';

router.get(`/${ROUTE_NAME}/`, controller.getTransaction.bind(controller));
export default router;
