import {EntityRepository, getConnection, Repository} from 'typeorm';
import {TelegramDatasetRequest} from '../entity/TelegramDatasetRequest';
import {TelegramDatasetRequestStatus} from '../entity/TelegramDatasetRequestStatus';
import {TelegramDatasetRequestStatus as DomainTgDataStatus} from '../../domain/RequestStatus';

@EntityRepository(TelegramDatasetRequest)
export class TelegramDatasetRequestRepository extends Repository<TelegramDatasetRequest> {
    public async findNonCancelledRequestsIds(chatId: number): Promise<number[]> {
        const connection = getConnection();

        const res = await connection
            .createQueryBuilder(TelegramDatasetRequest, 'req')
            .select('req.id', 'id')
            .leftJoin('req.status', 'req_status')
            .andWhere('req_status.request_status != :done', {done: DomainTgDataStatus.DONE})
            .andWhere('req_status.request_status != :cancelled', {cancelled: DomainTgDataStatus.CANCELLED})
            .andWhere('req.chat_id = :chat_id', {chat_id: chatId})
            .getRawMany();
        return res.map((val) => val.id);
    }

    public async cancelRequestByIds(ids: number[]): Promise<void> {
        const connection = getConnection();
        const cancelledStatus = await connection
            .manager
            .findOne(TelegramDatasetRequestStatus, {
                where: {request_status: DomainTgDataStatus.CANCELLED}
            });
        const updateEntities = [];
        for (const id of ids) {
            const entity = new TelegramDatasetRequest();
            entity.id = id;
            entity.status_id = cancelledStatus.id;
            updateEntities.push(entity);
        }
        await connection.manager.save(updateEntities);
    }

    public async findNonCancelledRequest(chatId: number): Promise<TelegramDatasetRequest> {
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(TelegramDatasetRequest, 'req')
            .addSelect('req_status')
            .leftJoinAndSelect('req.status', 'req_status')
            .leftJoinAndSelect('req.gender', 'gender')
            .where('req_status.request_status != :done', {done: DomainTgDataStatus.DONE})
            .andWhere('req_status.request_status != :cancelled', {cancelled: DomainTgDataStatus.CANCELLED})
            .andWhere('req.chat_id = :chat_id', {chat_id: chatId})
            .getOne();

        return res;
    }

    public async findChatIdByDatasetRequestId(datasetRequestId: number, userId: number): Promise<number | undefined> {
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(TelegramDatasetRequest, 'req')
            .select('req.chat_id', 'chat_id')
            .innerJoin('req.request', 'dataset_req')
            .where('dataset_req.id = :req_id', {req_id: datasetRequestId})
            .andWhere('dataset_req.user_id = :user_id', {user_id: userId})
            .execute();

        if (res.length == 0)
            return undefined;
        return res[0].chat_id;
    }
}
