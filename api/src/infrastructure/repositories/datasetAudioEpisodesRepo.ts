import {EntityRepository, getConnection, Repository} from 'typeorm';
import {DatasetAudioEpisodes} from '../entity/DatasetAudioEpisodes';

import {DatasetEpisodesTypes as DomainDatasetEpisodesTypes} from '../../domain/DatasetEpisodesTypes';

@EntityRepository(DatasetAudioEpisodes)
export class DatasetAudioEpisodesRepository extends Repository<DatasetAudioEpisodes> {
    public async findByAudioIdOrFail(audioId: number): Promise<DatasetAudioEpisodes[]> {
        return this.find({where: {audio_info_id: audioId}});
    }

    public async findIdsByAudioId(audioId: number): Promise<number[]> {
        const res = await this.find({select: ['id'], where: {audio_info_id: audioId}});
        return res.map((item) => item.id);
    }

    public async findIdsByAudioIdAndType(audioId: number, episodeType: DomainDatasetEpisodesTypes)
        : Promise<number[]> {
        const res = await getConnection()
            .createQueryBuilder(DatasetAudioEpisodes, 'episodes')
            .select('episodes.id', 'episode_id')
            .innerJoin('episodes.episode_type', 'episode_type')
            .where('episode_type.episode_type = :type', {type: episodeType})
            .andWhere('episodes.audio_info_id = :audio_id', {audio_id: audioId})
            .getRawMany();

        console.log('got this res');
        console.log(res);

        return res.map(object => object.episode_id);
    }
}
