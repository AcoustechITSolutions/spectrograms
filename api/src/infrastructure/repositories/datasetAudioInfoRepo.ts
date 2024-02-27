import {DatasetAudioTypes} from '../../domain/DatasetAudio';
import {EntityRepository, Repository, getConnection} from 'typeorm';

import {DatasetAudioInfo} from '../entity/DatasetAudioInfo';

type AudioPathDatabaseResponse = {
    record_path?: string
}

@EntityRepository(DatasetAudioInfo)
export class DatasetAudioInfoRepository extends Repository<DatasetAudioInfo> {
    public async findAudioIdByRequestIdOrFail(requestId: number, type: string): Promise<number> {
        const connection = getConnection();

        const res = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .select('audio.id', 'audio_id')
            .innerJoin('audio.audio_type', 'audio_type')
            .where('audio.request_id = :request_id', {request_id: requestId})
            .andWhere('audio_type.audio_type = :type', {type: type})
            .getRawOne();

        return res?.audio_id;
    }

    public async findAudioInfoRequestByIdOrFail(requestId: number, type: string): Promise<DatasetAudioInfo> {
        const connection = getConnection();

        const res = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .innerJoin('audio.audio_type', 'audio_type')
            .where('audio.request_id = :request_id', {request_id: requestId})
            .andWhere('audio_type.audio_type = :type', {type: type})
            .getOne();
        return res;
    }

    public async findAudioInfoRequestById(requestId: number, type: string): Promise<DatasetAudioInfo|undefined> {
        const connection = getConnection();

        const res = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .innerJoin('audio.audio_type', 'audio_type')
            .where('audio.request_id = :request_id', {request_id: requestId})
            .andWhere('audio_type.audio_type = :type', {type: type})
            .getOne();
        return res;
    }

    public async findAudioIdsByRequestId(requestId: number): Promise<number[]> {
        const connection = getConnection();

        const res = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .select('audio.id', 'audio_id')
            .where('audio.request_id = :request_id', {request_id: requestId})
            .getRawMany();

        return res.map(elem => elem.audio_id);

    }

    public async findAudioPathByRequestId(requestId: number, type: DatasetAudioTypes): Promise<string | undefined> {
        const connection = getConnection();
        const res = await connection
            .createQueryBuilder(DatasetAudioInfo, 'audio')
            .select('audio.audio_path', 'record_path')
            .leftJoin('audio.audio_type', 'audio_type')
            .where('audio.request_id = :req_id', {req_id: requestId})
            .andWhere('audio_type.audio_type = :type', {type: type})
            .getRawOne() as AudioPathDatabaseResponse;

        return res?.record_path;
    }
}
