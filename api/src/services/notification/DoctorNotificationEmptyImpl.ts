import {DoctorNotificationService} from './DoctorNotificationService';

export class DoctorNotificationEmptyImpl extends DoctorNotificationService {
    public async notifyAboutNewDiagnostic() {
        return Promise.resolve();
    }

    public async notifyAboutSupportRequest(contactData: string, userMessage: string, userId?: number) {
        return Promise.resolve();
    }
}
