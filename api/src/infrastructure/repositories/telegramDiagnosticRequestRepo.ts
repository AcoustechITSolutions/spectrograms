import {EntityRepository, getConnection, Repository} from 'typeorm';
import {BreathingDurationTypes} from '../entity/BreathingDurationTypes';
import {TelegramDiagnosticRequest} from '../entity/TelegramDiagnosticRequest';
import {TelegramDiagnosticRequestStatus} from '../entity/TelegramDiagnosticRequestStatus';
import {TelegramDiagnosticRequestStatus as DomainTgStatus} from '../../domain/RequestStatus';

@EntityRepository(TelegramDiagnosticRequest)
export class TelegramDiagnosticRequestRepository extends Repository<BreathingDurationTypes> {
    public async findNonCancelledRequestsIds(chatId: number): Promise<number[]> {
        const connection = getConnection();
            
        const res = await connection
            .createQueryBuilder(TelegramDiagnosticRequest, 'req')
            .select('req.id', 'id')
            .leftJoin('req.status', 'req_status')
            .andWhere('req_status.request_status != :done', {done: DomainTgStatus.DONE})
            .andWhere('req_status.request_status != :cancelled', {cancelled: DomainTgStatus.CANCELLED})
            .andWhere('req.chat_id = :chat_id', {chat_id: chatId})
            .getRawMany();
        return res.map((val) => val.id);
    }

    public async cancelRequestByIds(ids: number[]): Promise<void> {
        const connection = getConnection();
        const cancelledStatus = await connection
            .manager
            .findOne(TelegramDiagnosticRequestStatus, {
                where: {request_status: DomainTgStatus.CANCELLED}
            });
        const updateEntities = [];
        for (const id of ids) {
            const entity = new TelegramDiagnosticRequest();
            entity.id = id;
            entity.status_id = cancelledStatus.id;
            updateEntities.push(entity);
        }
        await connection.manager.save(updateEntities);
    }

    public async findNonCancelledRequest(chatId: number): Promise<TelegramDiagnosticRequest> {
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(TelegramDiagnosticRequest, 'req')
            .addSelect('req_status')
            .leftJoinAndSelect('req.status', 'req_status')
            .leftJoinAndSelect('req.gender', 'gender')
            .where('req_status.request_status != :done', {done: DomainTgStatus.DONE})
            .andWhere('req_status.request_status != :cancelled', {cancelled: DomainTgStatus.CANCELLED})
            .andWhere('req.chat_id = :chat_id', {chat_id: chatId})
            .getOne();

        return res;
    }

    public async findChatIdByDiagnosticRequestId(diagnosticRequestId: number, userId: number): Promise<number | undefined> {
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(TelegramDiagnosticRequest, 'req')
            .select('req.chat_id', 'chat_id')
            .innerJoin('req.request', 'diagnostic_req')
            .where('diagnostic_req.id = :req_id', {req_id: diagnosticRequestId})
            .andWhere('diagnostic_req.user_id = :user_id', {user_id: userId})
            .execute();
        
        if (res.length == 0)
            return undefined;
        return res[0].chat_id;
    }

    public async findDateByDiagnosticRequestId(diagnosticRequestId: number): Promise<string | undefined> {
        /**
        * Returns date and time when a request was created in HH:MI (DD.MM.YY) format, UTC+3 (Moscow).
        * Request creation time is a time of sending the data to the diagnostic_requests table.
        * This is needed for resulting report generation so that the users can keep track of their requests.
        */
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(TelegramDiagnosticRequest, 'req')
            .select('TO_CHAR(req.date_finished AT TIME ZONE \'MSK\', \'HH24:MI (DD.MM.YY)\')', 'date_finished')
            .innerJoin('req.request', 'diagnostic_req')
            .where('diagnostic_req.id = :req_id', {req_id: diagnosticRequestId})
            .execute();
        
        if (res.length == 0)
            return undefined;
        return res[0].date_finished;
    }
}
