import {EntityRepository, Repository, getConnection} from 'typeorm';
import {DatasetPatientDiseases} from '../entity/DatasetPatientDiseases';

@EntityRepository(DatasetPatientDiseases)
export class DatasetPatientDiseasesRepository extends Repository<DatasetPatientDiseases> {
    public async findDiseasesByRequestIdOrFail(requestId: number): Promise<DatasetPatientDiseases> {
        const connection = getConnection();

        const res = await connection.manager.findOneOrFail(DatasetPatientDiseases, {
            where: {
                request_id: requestId,
            },
        });

        return res;
    }
}
