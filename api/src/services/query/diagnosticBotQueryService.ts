import config from '../../config/config';
import {CoughAudio} from '../../infrastructure/entity/CoughAudio';
import {CoughCharacteristics} from '../../infrastructure/entity/CoughCharacteristic';
import {PatientInfo} from '../../infrastructure/entity/PatientInfo';
import {getConnection, getCustomRepository} from 'typeorm';
import {DiagnosticRequestStatusRepository} from '../../infrastructure/repositories/diagnosticRequestStatusRepo';
import {TelegramDiagnosticRequest} from '../../infrastructure/entity/TelegramDiagnosticRequest';
import {TgNewDiagnosticRequest} from '../../infrastructure/entity/TgNewDiagnosticRequest';
import {MuusDiagnosticRequest} from '../../infrastructure/entity/MuusDiagnosticRequest';
import {DiagnosticReport} from '../../infrastructure/entity/DiagnosticReport';
import {DiagnosticRequestStatus} from '../../domain/RequestStatus';
import {DiagnosticRequest} from '../../infrastructure/entity/DiagnosticRequest';
import {BotUserRepository} from '../../infrastructure/repositories/botUserRepo';

export const persistAsDiagnosticRequest = async (req: TelegramDiagnosticRequest|TgNewDiagnosticRequest|MuusDiagnosticRequest, userId: number) => {
    const statusRepo = getCustomRepository(DiagnosticRequestStatusRepository);
    const diagnosticRequest = new DiagnosticRequest();
    const processingStatus = await statusRepo.findByStringOrFail(DiagnosticRequestStatus.PROCESSING);
    const botUser = await getCustomRepository(BotUserRepository).findOneByChatId(req.chat_id);

    diagnosticRequest.user_id = userId;
    diagnosticRequest.status = processingStatus;
    diagnosticRequest.language = botUser.report_language;

    const patientInfo = new PatientInfo();
    patientInfo.request = diagnosticRequest;
    patientInfo.age = req.age;
    patientInfo.gender = req.gender.gender_type;
    patientInfo.is_smoking = req.is_smoking;
    patientInfo.sick_days = 0;

    const coughCharacteristic = new CoughCharacteristics();
    coughCharacteristic.request = diagnosticRequest;
    if ('is_forced' in req) {
        coughCharacteristic.is_forced = req.is_forced;
    } else {
        coughCharacteristic.is_forced = false;
    }

    const coughAudio = new CoughAudio();
    coughAudio.request = diagnosticRequest;
    coughAudio.file_path = req.cough_audio_path;

    const connection = getConnection();
    const queryRunner = connection.createQueryRunner();
    await queryRunner.startTransaction();

    const manager = queryRunner.manager;
    try {
        await manager.save(diagnosticRequest, {transaction: false});
        const diagnosticReport = new DiagnosticReport();
        diagnosticReport.user_id = userId;
        diagnosticReport.request = diagnosticRequest;
        req.request = diagnosticRequest;

        await manager.save([diagnosticReport, req, patientInfo, coughCharacteristic, coughAudio], 
            {transaction: false});
        await queryRunner.commitTransaction();
        return diagnosticRequest.id;
    } catch(error) {
        console.error(error);
        queryRunner.rollbackTransaction();
    } finally {
        queryRunner.release();
    }
};
