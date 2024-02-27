/**
Service for notifying doctors about new diagnostic requests.
 */
export abstract class DoctorNotificationService {
    /**
     * Message about new diagnostic. Should stay the same in implementation.
     */
    protected readonly NEW_DIAGNOSTIC_MESSAGE = 'Поступили новые записи для обработки.';
    /**
     * Should notify all subscribed doctors about new diagnostic requests.
     */
    abstract notifyAboutNewDiagnostic(): Promise<void>
    /**
     * Should notify all subscribed doctors about support requests.
     */
    abstract notifyAboutSupportRequest(contactData: string, userMessage: string, userId?: number): Promise<void>
}