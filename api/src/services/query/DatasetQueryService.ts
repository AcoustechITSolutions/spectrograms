import {PaginationParams} from '../../interfaces/PaginationParams';
import {getConnection} from 'typeorm';
import {DatasetRequest} from '../../infrastructure/entity/DatasetRequest';

export type UserDatasetRecord = {
    readonly date: Date,
    readonly id: number,
    readonly identifier: string
}

/**
 * Queries read-only view models.
 */
export class DatasetQueryService {
    /**
     * @param connection {Connection} - typeorm database connection.
     */
    constructor() { // FIXME: inject connection through di container.
    }

    public async fetchDatasetRecordByUserId(userId: number, paginationParams: PaginationParams) {
        const connection = getConnection();
        return connection
            .createQueryBuilder(DatasetRequest, 'req')
            .select('req.id', 'id')
            .addSelect('req.date_created', 'date')
            .addSelect('patient.identifier', 'identifier')
            .innerJoin('dataset_patient_details', 'patient', 'patient.request_id = req.id')
            .where('req.user_id = :userId', {userId: userId})
            .orderBy('req.date_created', 'DESC')
            .offset(paginationParams.offset)
            .limit(paginationParams.limit)
            .execute() as Promise<UserDatasetRecord[]>;
    }    
}
