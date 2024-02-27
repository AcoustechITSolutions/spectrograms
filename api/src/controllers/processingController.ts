import {Request, Response} from 'express';
import {HttpStatusCodes, getErrorMessage, HttpErrors} from '../helpers/status';
import {DiagnosticRequest} from '../infrastructure/entity/DiagnosticRequest';
import {DiagnosticRequestStatus} from '../infrastructure/entity/DiagnostRequestStatus';
import {DiagnosticRequestStatus as DomainDiagnosticRequestStatus} from '../domain/RequestStatus';
import {getRepository, getConnection, getManager, getCustomRepository} from 'typeorm';
import {CoughAudio} from '../infrastructure/entity/CoughAudio';
import {PatientInfo} from '../infrastructure/entity/PatientInfo';
import {getFileExtName} from '../helpers/file';
import {DiagnosticReport} from '../infrastructure/entity/DiagnosticReport';
import {CoughCharacteristics} from '../infrastructure/entity/CoughCharacteristic';
import {DiagnosisTypes as DiagnosisTypesEntity} from '../infrastructure/entity/DiagnosisTypes';
import {DiagnosisTypes as DomainDiagnosisTypes} from '../domain/DiagnosisTypes';
import {fileService} from '../container';
import {CoughIntensityTypes} from '../infrastructure/entity/CoughIntensityTypes';
import {CoughProductivityTypes} from '../infrastructure/entity/CoughProductivityTypes';
import {fetchProcessingGeneral, AgeFilter, ProcessingFilterParams, fetchProcessingById, fetchProcessingNavigationByRequestId, fetchDiagnosisStatistics} from '../services/query/processingQueryService';
import {DiagnosticRequestStatusRepository} from '../infrastructure/repositories/diagnosticRequestStatusRepo';
import {createSortByRegExp, getSortingParamsByRegExp} from '../helpers/sort';
import {Gender} from '../domain/Gender';
import {onNoisyRecord, onPatchDiagnostic} from '../services/diagnostic/diagnosticBotNotificationService';
import {onNewNoisyRecord, onPatchNewDiagnostic} from '../services/diagnostic/newDiagnosticBotNotificationService';
import {onMuusNoisyRecord, onPatchMuusDiagnostic} from '../services/diagnostic/muusBotNotificationService';
import {DiagnosticRequestRepository} from '../infrastructure/repositories/diagnosticRequestRepo';
import {NavigationParams} from '../interfaces/NavigationParams';
import {UserRepository} from '../infrastructure/repositories/userRepo';
import {DiagnosticReportEpisodes} from '../infrastructure/entity/DiagnosisReportEpisodes';
import {PayonlineTransactions} from '../infrastructure/entity/PayonlineTransactions';
import {TelegramDiagnosticRequest} from '../infrastructure/entity/TelegramDiagnosticRequest';
import {TgNewDiagnosticRequest} from '../infrastructure/entity/TgNewDiagnosticRequest';
import {MuusDiagnosticRequest} from '../infrastructure/entity/MuusDiagnosticRequest';
import {UserRoleTypes} from '../domain/UserRoles';

interface GetProcessingResponse {
    number: number,
    status: string,
    diagnosis: string,
    request_id: number,
    date_created: Date,
    age: number,
    gender: string,
    identifier: string,
    nationality: string,
    is_pcr_positive: boolean
}

interface GetProcessingByIdResponse {
    intensity: string,
    productivity: string,
    diagnosis: string,
    diagnosis_probability: number,
    commentary: string,
    date_created: Date,
    status: string,
    language?: string,
    identifier?: string,
    nationality?: string,
    is_pcr_positive?: boolean,
    navigation: NavigationParams
}

interface PatchDetailsBody {
    nationality?: string,
    is_pcr_positive?: boolean,
    identifier?: string,
    commentary?: string
}

interface GetStatisticResponse {
    healthy: number,
    at_risk: number,
    covid_19: number
}

const PROCESSING_SORTING_FIELDS = ['age', 'gender', 'date_created', 'status', 'diagnosis', 'request_id', 'identifier', 'nationality', 'is_pcr_positive'];
const PROCESSING_REGEXP = createSortByRegExp(PROCESSING_SORTING_FIELDS);

/**
 * Responsible for handling requests to /admin/processing routes. 
 * This route is intended to provide doctors interface to diagnostic records.
 */

export class ProcessingController { // TODO: specify dependencies that must be injected

    private async parseProcessingFilters(req: Request) {
        let sourceFilter = new Array<string>();
        const userRepo = getCustomRepository(UserRepository);
        const doctorUser = await userRepo.findOne(req.token.userId);
        if (!doctorUser.is_all_patients) {
            sourceFilter = await userRepo.findPatientsByUserId(req.token.userId);
            if (sourceFilter.length == 0) {
                throw new Error(HttpErrors.NO_PATIENTS);
            }
        }
        const age = req.query.age as any;
        const ageFilter: AgeFilter = {
            lte: isNaN(age?.lte) ? undefined : Number(age.lte),
            gte: isNaN(age?.gte) ? undefined : Number(age.gte)
        };
        const statusFilterParams = req.query.status;
        const diagnosisFilterParams = req.query.diagnosis;
        const genderFilterParams = req.query.gender;
        const statusFilter = new Array<DomainDiagnosticRequestStatus>();
        const diagnosisFilter = new Array<DomainDiagnosisTypes>();
        const genderFilter = new Array<Gender>();
        if (Array.isArray(genderFilterParams)) {
            for (const gender of genderFilterParams) {
                const type = Object.values(Gender).find(val => val == gender);
                if (type == undefined) {
                    throw new Error(HttpErrors.NO_GENDER);
                }
                genderFilter.push(type);
            }
        } else if (genderFilterParams != undefined) {
            const type = Object.values(Gender).find(val => val == genderFilterParams);
            if (type == undefined) {
                throw new Error(HttpErrors.NO_GENDER);
            }
            genderFilter.push(type);
        }

        if (Array.isArray(statusFilterParams)) {
            for (const filterStatus of statusFilterParams) {
                const type = Object.values(DomainDiagnosticRequestStatus).find(val => val == filterStatus);
                if (type == undefined) {
                    throw new Error(HttpErrors.NO_STATUS);
                }
                statusFilter.push(type);
            }
        } else if (statusFilterParams != undefined) {
            const type = Object.values(DomainDiagnosticRequestStatus).find(val => val == statusFilterParams);
            if (type == undefined) {
                throw new Error(HttpErrors.NO_STATUS);
            }
            statusFilter.push(type);
        }

        if (Array.isArray(diagnosisFilterParams)) {
            for (const filterDiagnosis of diagnosisFilterParams) {
                const type = Object.values(DomainDiagnosisTypes).find(val => val == filterDiagnosis);
                if (type == undefined) {
                    throw new Error(HttpErrors.NO_DIAGNOSIS);
                }
                diagnosisFilter.push(type);
            }
        } else if (diagnosisFilterParams != undefined) {
            const type =  Object.values(DomainDiagnosisTypes).find(val => val == diagnosisFilterParams);
            if (type == undefined) {
                throw new Error(HttpErrors.NO_DIAGNOSIS);
            }
            diagnosisFilter.push(type);
        }

        let roleFilter: UserRoleTypes;
        if (req.token.roles?.includes(UserRoleTypes.VIEWER)) {
            roleFilter = UserRoleTypes.VIEWER;
        } else {
            roleFilter = UserRoleTypes.EDIFIER;
        }

        return Promise.resolve({
            roleFilter: roleFilter,
            sourceFilter: sourceFilter,
            statusFilter: statusFilter,
            diagnosisFilter: diagnosisFilter,
            genderFilter: genderFilter,
            ageFilter: ageFilter
        })
    }

    public async getProcessing (req: Request, res: Response) {
        try {
            let filterParams: ProcessingFilterParams;
            try {
                filterParams = await this.parseProcessingFilters(req);
            } catch (error) {
                const errorMessage = getErrorMessage(error.message);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
         
            const sortingParams = getSortingParamsByRegExp(req.query.sort_by as string, PROCESSING_REGEXP);
            const processingRecords = await fetchProcessingGeneral(req.paginationParams, filterParams, sortingParams);
    
            const response: GetProcessingResponse[] = processingRecords.map((record) => {
                return {
                    number: record.serial_number,
                    status: record.status,
                    diagnosis: record.diagnosis,
                    request_id: record.request_id,
                    age: record.age,
                    date_created: record.date_created,
                    gender: record.gender,
                    identifier: record.identifier,
                    nationality: record.nationality,
                    is_pcr_positive: record.is_pcr_positive,
                };
            });
    
            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }
    
    public async getProcessingById (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const sortingParams = getSortingParamsByRegExp(req.query.sort_by as string, PROCESSING_REGEXP);
        let filterParams: ProcessingFilterParams;
        try {
            filterParams = await this.parseProcessingFilters(req);
        } catch (error) {
            const errorMessage = getErrorMessage(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
      
        try {
            const requests = await Promise.all([
                fetchProcessingNavigationByRequestId(requestId, filterParams, sortingParams),
                fetchProcessingById(requestId)
            ]);
            const navigation = requests[0];
            const data = requests[1];
            if (navigation == undefined) {
                const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
                return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
            }

            const response: GetProcessingByIdResponse = {
                intensity: data.intensity,
                productivity: data.productivity,
                diagnosis: data.diagnosis,
                diagnosis_probability: data.diagnosis_probability,
                commentary: data.commentary,
                date_created: data.date_created,
                status: data.status,
                language: data.language,
                identifier: data.identifier,
                nationality: data.nationality,
                is_pcr_positive: data.is_pcr_positive,
                navigation: navigation
            };

            return res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getProcessingStatistics (req: Request, res: Response) {
        let filterParams: ProcessingFilterParams;
        try {
            filterParams = await this.parseProcessingFilters(req);
        } catch (error) {
            const errorMessage = getErrorMessage(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        try {
            const statistics = await fetchDiagnosisStatistics(filterParams);     
    
            const response: GetStatisticResponse = {
                healthy: statistics.find(obj => {return obj.category == DomainDiagnosisTypes.HEALTHY})?.value ?? 0,
                at_risk: statistics.find(obj => {return obj.category == DomainDiagnosisTypes.AT_RISK})?.value ?? 0,
                covid_19: statistics.find(obj => {return obj.category == DomainDiagnosisTypes.COVID_19})?.value ?? 0
            };
            res.status(HttpStatusCodes.SUCCESS).send(response);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }
    
    public async patchProcessing (req: Request, res: Response) {
        const requestId = req.params.id;
        const diagnosticReportRepo = getRepository(DiagnosticReport);
    
        const {diagnosis, diagnosis_probability, commentary,
            intensity, productivity} = req.body;
    
        let diagnosticReport: DiagnosticReport;
        try {
            diagnosticReport = await diagnosticReportRepo.findOneOrFail({
                relations: ['request', 'request.status'],
                where: {
                    request_id: requestId,
                },
            });
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
    
        if (diagnosticReport.request.status.request_status == DomainDiagnosticRequestStatus.ERROR 
            || diagnosticReport.request.status.request_status == DomainDiagnosticRequestStatus.NOISY_AUDIO) {
                const errorMessage = getErrorMessage(HttpErrors.PATCH_RECORD_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        const connection = getConnection();
    
        diagnosticReport.is_confirmed = true;
        diagnosticReport.commentary = commentary == undefined ? '' : commentary;

        let diagnosisType: DiagnosisTypesEntity;
        if (diagnosis != undefined) {
            try {
                diagnosisType = await connection.manager.findOneOrFail(DiagnosisTypesEntity, {
                    where: {diagnosis_type: diagnosis as string},
                });
            } catch(error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_DIAGNOSIS);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }
    
        diagnosticReport.diagnosis_id = diagnosis == undefined ? diagnosticReport.diagnosis_id : diagnosisType.id;
        diagnosticReport.diagnosis_probability = diagnosis_probability == undefined ? diagnosticReport.diagnosis_probability : Number(diagnosis_probability);
    
        let intensityType: CoughIntensityTypes;
        if (intensity != undefined) {
            try {
                intensityType = await connection.manager.findOneOrFail(CoughIntensityTypes, {
                    where: {intensity_type: intensity as string},
                });
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }
    
        let productivityType: CoughProductivityTypes;
        if (productivity != undefined) {
            try {
                productivityType = await connection.manager.findOneOrFail(CoughProductivityTypes, {
                    where: {productivity_type: productivity as string},
                });
            } catch (error) {
                console.error(error);
                const errorMessage = getErrorMessage(HttpErrors.NO_COUGH_CHARACTERISTIC);
                return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
            }
        }
    
        let coughCharacteristic: CoughCharacteristics;
        try {
            coughCharacteristic = await getRepository(CoughCharacteristics).findOneOrFail({
                where: {
                    request_id: requestId,
                },
            });
    
            coughCharacteristic.intensity_id = intensity == undefined ? coughCharacteristic.intensity_id : intensityType.id;
            coughCharacteristic.productivity_id = productivity == undefined ? coughCharacteristic.productivity_id : productivityType.id;
    
            const statusRepo = getCustomRepository(DiagnosticRequestStatusRepository);
            const successStatus = await statusRepo.findByStringOrFail(DomainDiagnosticRequestStatus.SUCCESS);
            diagnosticReport.request.status = successStatus;
            await getManager().save([diagnosticReport, coughCharacteristic, diagnosticReport.request]);
        } catch (error) {
            console.error(error);
            return res.status(HttpStatusCodes.BAD_REQUEST).send();
        }
        
        onPatchDiagnostic(Number(requestId));
        onPatchNewDiagnostic(Number(requestId));
        onPatchMuusDiagnostic(Number(requestId));
        return res.status(HttpStatusCodes.SUCCESS).send();
    }

    public async patchProcessingDetails (req: Request, res: Response) {
        const requestId = req.params.id;
        const diagnosticReportRepo = getRepository(DiagnosticReport);
        const patientInfoRepo = getRepository(PatientInfo);
        let requestBody: PatchDetailsBody;
        try {
            requestBody = {
                nationality: req.body.nationality,
                is_pcr_positive: req.body.is_pcr_positive,
                identifier: req.body.identifier,
                commentary: req.body.commentary
            };
        } catch(error) {
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        let diagnosticReport: DiagnosticReport;
        let patientInfo: PatientInfo;
        try {
            diagnosticReport = await diagnosticReportRepo.findOneOrFail({
                relations: ['request', 'request.status'],
                where: {
                    request_id: requestId
                }
            });
            patientInfo = await patientInfoRepo.findOneOrFail({
                where: {
                    request_id: requestId
                }
            });
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
    
        if (diagnosticReport.request.status.request_status == DomainDiagnosticRequestStatus.ERROR 
            || diagnosticReport.request.status.request_status == DomainDiagnosticRequestStatus.NOISY_AUDIO) {
                const errorMessage = getErrorMessage(HttpErrors.PATCH_RECORD_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        diagnosticReport.is_pcr_positive = requestBody.is_pcr_positive;
        diagnosticReport.nationality = requestBody.nationality;
        diagnosticReport.commentary = requestBody.commentary;
        patientInfo.identifier = requestBody.identifier;
        
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;
        try {
            await manager.save([diagnosticReport, patientInfo]);
            await queryRunner.commitTransaction()
            return res.status(HttpStatusCodes.SUCCESS).send();
        } catch (error) {
            console.error(error);
            await queryRunner.rollbackTransaction();
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        } finally {
            await queryRunner.release();
        }
    }
    
    public async getProcessingCoughAudio (req: Request, res: Response) {
        const requestId = req.params.id;
    
        const coughAudioRepo = getRepository(CoughAudio);
        let coughAudio: CoughAudio;
        try {
            coughAudio = await coughAudioRepo.findOneOrFail({
                select: ['file_path'],
                where: {
                    request_id: requestId,
                },
            });
        } catch (error) {
            console.error(error);
            res.setHeader('Content-Type', 'application/json');
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        res.status(HttpStatusCodes.SUCCESS);
        const extension = getFileExtName(coughAudio.file_path);
        res.setHeader('Content-Type', `audio/${extension}`);
        try {
            const stream = await fileService
                .getFileAsStream(coughAudio.file_path);
    
            stream
                .on('error', (err) => {
                    console.error(err);
                    const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                    res.setHeader('Content-Type', 'application/json');
                    return res.status(HttpStatusCodes.ERROR).send(errorMessage);
                })
                .pipe(res);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getProcessingSpectrogram (req: Request, res: Response) {
        const requestId = Number(req.params.id);
    
        const coughAudioRepo = getRepository(CoughAudio);
        let coughAudio: CoughAudio;
        try {
            coughAudio = await coughAudioRepo.findOneOrFail({
                select: ['spectrogram_path'],
                where: {
                    request_id: requestId
                }
            });
        } catch (error) {
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        res.status(HttpStatusCodes.SUCCESS);
        res.setHeader('Content-Type', 'image/png');
        try {
            const stream = await fileService
                .getFileAsStream(coughAudio.spectrogram_path);
         
            stream.on('error', (err) => {
                console.error(err);
                const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                return res.status(HttpStatusCodes.ERROR).send(errorMessage);
            })
                .pipe(res);
        } catch (error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    public async getProcessingPatientPhoto (req: Request, res: Response) {
        const requestId = req.params.id;
    
        const PatientInfoRepo = getRepository(PatientInfo);
        let patientInfo: PatientInfo;
        try {
            patientInfo = await PatientInfoRepo.findOneOrFail({
                select: ['photo_path'],
                where: {
                    request_id: requestId,
                },
            });
        } catch (error) {
            console.error(error);
            res.setHeader('Content-Type', 'application/json');
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
    
        res.status(HttpStatusCodes.SUCCESS);
        res.setHeader('Content-Type', 'image/*');
        try {
            const stream = await fileService.getFileAsStream(patientInfo.photo_path);
    
            stream
                .on('error', (err) => {
                    console.error(err);
                    const errorMessage = getErrorMessage(HttpErrors.FILE_SENDING_ERROR);
                    res.setHeader('Content-Type', 'application/json');
                    return res.status(HttpStatusCodes.ERROR).send(errorMessage);
                })
                .pipe(res);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
    }

    /**
     * Change diagnostic request status to noisy_audio. 
     * If request was from the telegram bot, user will be notified about this.
     * @param req {Request} - express request
     * @param res {Response} - express response
     */
    public async markRecordNoisy (req: Request, res: Response) {
        const requestId = Number(req.params.id);
        const requestRepo = getCustomRepository(DiagnosticRequestRepository);
        const statusRepo = getCustomRepository(DiagnosticRequestStatusRepository);

        let diagnosticRequest: DiagnosticRequest;
        try {
            diagnosticRequest = await requestRepo
                .findRequestByIdOrFail(requestId);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        }
        
        let noisyStatus: DiagnosticRequestStatus;
        try {
            noisyStatus = await statusRepo
                .findByStringOrFail(DomainDiagnosticRequestStatus.NOISY_AUDIO);  
            diagnosticRequest.status_id = noisyStatus.id;   
            await requestRepo.save(diagnosticRequest);
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
        
        onNoisyRecord(requestId);
        onNewNoisyRecord(requestId);
        onMuusNoisyRecord(requestId);
        return res.status(HttpStatusCodes.NO_CONTENT).send();
    }

    public async deleteProcessing (req: Request, res: Response) {
        const requestId = Number(req.params.id);
    
        const connection = getConnection();
        const queryRunner = connection.createQueryRunner();
        await queryRunner.startTransaction();
        const manager = queryRunner.manager;

        try {
            const diagnosticReport = manager.findOneOrFail(DiagnosticReport, {select: ['id'], where: {request_id: requestId}});
            const coughAudio = manager.findOneOrFail(CoughAudio, {select: ['file_path'], where: {request_id: requestId}});
            const patientInfo = manager.findOne(PatientInfo, {select: ['photo_path'], where: {request_id: requestId}});
            const telegramBotData = manager.findOne(TelegramDiagnosticRequest, {select: ['cough_audio_path'], where: {request_id: requestId}});
            const tgNewBotData = manager.findOne(TgNewDiagnosticRequest, {select: ['cough_audio_path'], where: {request_id: requestId}});
            const muusBotData = manager.findOne(MuusDiagnosticRequest, {select: ['cough_audio_path'], where: {request_id: requestId}});
           
            const dbResult = await Promise.all([diagnosticReport, coughAudio, patientInfo, telegramBotData, tgNewBotData, muusBotData]);
            const report = dbResult[0];
            const cough = dbResult[1];
            const patient = dbResult[2];
            const telegramBot = dbResult[3];
            const tgNewBot = dbResult[4];
            const muusBot = dbResult[5];

            if (cough?.file_path != undefined) {
                await fileService.deleteDirectory(cough.file_path);
            }
            if (patient?.photo_path != undefined) {
                await fileService.deleteDirectory(patient.photo_path);
            }
            if (telegramBot?.cough_audio_path != undefined) {
                await fileService.deleteDirectory(telegramBot.cough_audio_path);
            }
            if (tgNewBot?.cough_audio_path != undefined) {
                await fileService.deleteDirectory(tgNewBot.cough_audio_path);
            }
            if (muusBot?.cough_audio_path != undefined) {
                await fileService.deleteDirectory(muusBot.cough_audio_path);
            }
            
            await manager.delete(DiagnosticReportEpisodes, {report_id: report.id});
            await manager.delete(DiagnosticReport, {request_id: requestId});
            await manager.delete(CoughAudio, {request_id: requestId});
            await manager.delete(CoughCharacteristics, {request_id: requestId});
            await manager.delete(PatientInfo, {request_id: requestId});
            await manager.delete(PayonlineTransactions, {request_id: requestId});
            await manager.delete(TelegramDiagnosticRequest, {request_id: requestId});
            await manager.delete(TgNewDiagnosticRequest, {request_id: requestId});
            await manager.delete(MuusDiagnosticRequest, {request_id: requestId});
            await manager.delete(DiagnosticRequest, {id: requestId});

            await queryRunner.commitTransaction();
            return res.status(HttpStatusCodes.SUCCESS).send({
                is_deleted: true,
            });
        } catch (error) {
            console.error(error);
            await queryRunner.rollbackTransaction();
            const errorMessage = getErrorMessage(HttpErrors.NO_RECORD);
            return res.status(HttpStatusCodes.NOT_FOUND).send(errorMessage);
        } finally {
            await queryRunner.release();
        }
    }
}
