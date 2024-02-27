/* eslint @typescript-eslint/no-var-requires: "off" */
const HL7 = require('hl7-standard');
import moment from 'moment';
import {DiagnosticRequestStatus} from '../domain/RequestStatus';

const APP_NAME = 'Acoustery';

type DiagnosticInfo = {
    requestId: number,
    dateCreated: Date,
    status: DiagnosticRequestStatus,
    diagnosis?: string,
    probability?: number,
    commentary?: string,
    userId: number,
    userLogin: string,
    age?: number,
    gender?: string,
    identifier?: string,
    nationality?: string,
    isPcrPositive?: boolean,
    intensity?: string,
    productivity?: string,
}

export const generateDiagnosticHL7 = (requestInfo: DiagnosticInfo) => {
    const currentDate = new Date();
    const formattedCurrentDate = moment(currentDate).format('YYYYMMDDHHmmss');
    const formattedRequestDate = moment(requestInfo.dateCreated).format('YYYYMMDDHHmmss');
    let birthDate: string;
    if (requestInfo.age != undefined) {
        birthDate = moment(currentDate).subtract(requestInfo.age, 'y').format('YYYY');
    }
    let gender: string;
    if (requestInfo.gender != undefined) {
        gender = requestInfo.gender == 'male' ? 'M' : 'F';
    }
    const statusCode = new Map<DiagnosticRequestStatus, string>([
        [DiagnosticRequestStatus.PENDING, 'R'],  // not verified
        [DiagnosticRequestStatus.SUCCESS, 'F'],  // final
        [DiagnosticRequestStatus.ERROR, 'X'],  // canceled
        [DiagnosticRequestStatus.NOISY_AUDIO, 'X'],  // canceled
    ]);
    const status = statusCode.get(requestInfo.status);
    
    let hl7 = new HL7();
    hl7.createSegment('MSH');
    hl7.set('MSH', {
        'MSH.2': '^~\\&',
        'MSH.3': APP_NAME,
        'MSH.4': APP_NAME,
        'MSH.5': '', 'MSH.6': '',
        'MSH.7': formattedCurrentDate,
        'MSH.8': '',
        'MSH.9': {
            'MSH.9.1': 'ORU',
            'MSH.9.2': 'R01',
            'MSH.9.3': 'ORU_R01'
        },
        'MSH.10': requestInfo.requestId,
        'MSH.11': 'P',
        'MSH.12': '2.8',
        'MSH.13': '', 'MSH.14': '', 'MSH.15': '', 'MSH.16': '', 'MSH.17': '', 'MSH.18': '', 'MSH.19': '', 'MSH.20': '', 'MSH.21': '', 'MSH.22': '', 
        'MSH.23': '', 'MSH.24': ''
    });

    hl7.createSegment('PID');
    hl7.set('PID', {
        'PID.1': '', 'PID.2': '',
        'PID.3': [{
            'PID.3.1': requestInfo.userId,
            'PID.3.2': '', 'PID.3.3': '',
            'PID.3.4': APP_NAME,
            'PID.3.5': 'PI',
            'PID.3.6': '', 'PID.3.7': '', 'PID.3.8': '', 'PID.3.9': '', 'PID.3.10': '', 'PID.3.11': '', 'PID.3.12': ''
        }, {
            'PID.3.1': requestInfo.userLogin,
            'PID.3.2': '', 'PID.3.3': '',
            'PID.3.4': APP_NAME,
            'PID.3.5': 'PT',
            'PID.3.6': '', 'PID.3.7': '', 'PID.3.8': '', 'PID.3.9': '', 'PID.3.10': '', 'PID.3.11': '', 'PID.3.12': ''
        }],
        'PID.4': '',
        'PID.5': requestInfo.identifier ?? '-',
        'PID.6': '',
        'PID.7': birthDate ?? '',
        'PID.8': gender ?? '',
        'PID.9': '', 'PID.10': '', 'PID.11': '', 'PID.12': '', 'PID.13': '', 'PID.14': '', 'PID.15': '', 'PID.16': '', 'PID.17': '', 'PID.18': '', 
        'PID.19': '', 'PID.20': '', 'PID.21': '', 'PID.22': '', 
        'PID.23': requestInfo.nationality ?? '', 
        'PID.24': '', 'PID.25': '', 'PID.26': '', 'PID.27': '', 'PID.28': '', 'PID.29': '', 'PID.30': '', 'PID.31': '', 'PID.32': '', 'PID.33': '', 
        'PID.34': '', 'PID.35': '', 'PID.36': '', 'PID.37': '', 'PID.38': '', 'PID.39': ''
    });

    hl7.createSegment('OBR');
    hl7.set('OBR', {
        'OBR.1': '', 'OBR.2': '', 
        'OBR.3': requestInfo.requestId,
        'OBR.4': {
            'OBX.4.1': 'diagnostic_report',
            'OBX.4.2': 'Acoustery diagnostic report',
            'OBX.4.3': 'L'
        },
        'OBR.5': '', 'OBR.6': '', 
        'OBR.7': formattedRequestDate,
        'OBR.8': '', 'OBR.9': '', 'OBR.10': '', 'OBR.11': '', 'OBR.12': '', 'OBR.13': '', 'OBR.14': '', 'OBR.15': '', 'OBR.16': '', 'OBR.17': '', 
        'OBR.18': '', 'OBR.19': '', 'OBR.20': '', 'OBR.21': '', 'OBR.22': '', 'OBR.23': '', 'OBR.24': '', 
        'OBR.25': status,
        'OBR.26': '', 'OBR.27': '', 'OBR.28': '', 'OBR.29': '', 'OBR.30': '', 'OBR.31': '', 'OBR.32': '', 'OBR.33': '', 'OBR.34': '', 'OBR.35': '', 
        'OBR.36': '', 'OBR.37': '', 'OBR.38': '', 'OBR.39': '', 'OBR.40': '', 'OBR.41': '', 'OBR.42': '', 'OBR.43': '', 'OBR.44': '', 'OBR.45': '',
        'OBR.46': '', 'OBR.47': '', 'OBR.48': '', 'OBR.49': '', 'OBR.50': '', 'OBR.51': '', 'OBR.52': '', 'OBR.53': ''
    });

    let resultNumber = 0;
    if (requestInfo.diagnosis) {
        resultNumber += 1;
        hl7.createSegment('OBX');
        hl7.set('OBX', {
            'OBX.1': resultNumber, 
            'OBX.2': 'ST', 
            'OBX.3': {
                'OBX.3.1': 'diagnosis',
                'OBX.3.2': 'COVID-19 prediction',
                'OBX.3.3': 'L'
            }, 
            'OBX.4': '', 
            'OBX.5': requestInfo.diagnosis,
            'OBX.6': '', 'OBX.7': '', 'OBX.8': '',
            'OBX.9': requestInfo.probability.toFixed(3),
            'OBX.10': '',
            'OBX.11': status,
            'OBX.12': '', 'OBX.13': '',
            'OBX.14': formattedRequestDate,
            'OBX.15': '', 'OBX.16': '', 'OBX.17': '', 'OBX.18': '', 'OBX.19': '', 'OBX.20': '', 'OBX.21': '', 'OBX.22': '', 'OBX.23': '', 'OBX.24': '',
            'OBX.25': '', 'OBX.26': '', 'OBX.27': ''
        }, resultNumber-1);
    }
    
    if (requestInfo.productivity) {
        resultNumber += 1;
        hl7.createSegment('OBX');
        hl7.set('OBX', {
            'OBX.1': resultNumber, 
            'OBX.2': 'ST', 
            'OBX.3': {
                'OBX.3.1': 'productivity',
                'OBX.3.2': 'Cough productivity',
                'OBX.3.3': 'L'
            }, 
            'OBX.4': '', 
            'OBX.5': requestInfo.productivity,
            'OBX.6': '', 'OBX.7': '', 'OBX.8': '', 'OBX.9': '', 'OBX.10': '',
            'OBX.11': status,
            'OBX.12': '', 'OBX.13': '',
            'OBX.14': formattedRequestDate,
            'OBX.15': '', 'OBX.16': '', 'OBX.17': '', 'OBX.18': '', 'OBX.19': '', 'OBX.20': '', 'OBX.21': '', 'OBX.22': '', 'OBX.23': '', 'OBX.24': '',
            'OBX.25': '', 'OBX.26': '', 'OBX.27': ''
        }, resultNumber-1);
    }

    if (requestInfo.intensity) {
        resultNumber += 1;
        hl7.createSegment('OBX');
        hl7.set('OBX', {
            'OBX.1': resultNumber, 
            'OBX.2': 'ST', 
            'OBX.3': {
                'OBX.3.1': 'intensity',
                'OBX.3.2': 'Cough intensity',
                'OBX.3.3': 'L'
            }, 
            'OBX.4': '', 
            'OBX.5': requestInfo.intensity,
            'OBX.6': '', 'OBX.7': '', 'OBX.8': '', 'OBX.9': '', 'OBX.10': '',
            'OBX.11': status,
            'OBX.12': '', 'OBX.13': '',
            'OBX.14': formattedRequestDate,
            'OBX.15': '', 'OBX.16': '', 'OBX.17': '', 'OBX.18': '', 'OBX.19': '', 'OBX.20': '', 'OBX.21': '', 'OBX.22': '', 'OBX.23': '', 'OBX.24': '',
            'OBX.25': '', 'OBX.26': '', 'OBX.27': ''
        }, resultNumber-1);
    }

    if (requestInfo.commentary && requestInfo.commentary != '') {
        resultNumber += 1;
        hl7.createSegment('OBX');
        hl7.set('OBX', {
            'OBX.1': resultNumber, 
            'OBX.2': 'ST', 
            'OBX.3': {
                'OBX.3.1': 'commentary',
                'OBX.3.2': 'Comment',
                'OBX.3.3': 'L'
            }, 
            'OBX.4': '', 
            'OBX.5': requestInfo.commentary,
            'OBX.6': '', 'OBX.7': '', 'OBX.8': '', 'OBX.9': '', 'OBX.10': '',
            'OBX.11': status,
            'OBX.12': '', 'OBX.13': '',
            'OBX.14': formattedRequestDate,
            'OBX.15': '', 'OBX.16': '', 'OBX.17': '', 'OBX.18': '', 'OBX.19': '', 'OBX.20': '', 'OBX.21': '', 'OBX.22': '', 'OBX.23': '', 'OBX.24': '',
            'OBX.25': '', 'OBX.26': '', 'OBX.27': ''
        }, resultNumber-1);
    }

    if (requestInfo.isPcrPositive != undefined) {
        resultNumber += 1;
        hl7.createSegment('OBX');
        hl7.set('OBX', {
            'OBX.1': resultNumber, 
            'OBX.2': 'ST', 
            'OBX.3': {
                'OBX.3.1': 'is_pcr_positive',
                'OBX.3.2': 'COVID-19 PCR test',
                'OBX.3.3': 'L'
            }, 
            'OBX.4': '', 
            'OBX.5': requestInfo.isPcrPositive ? 'true' : 'false',
            'OBX.6': '', 'OBX.7': '', 'OBX.8': '', 'OBX.9': '', 'OBX.10': '',
            'OBX.11': status,
            'OBX.12': '', 'OBX.13': '',
            'OBX.14': formattedRequestDate,
            'OBX.15': '', 'OBX.16': '', 'OBX.17': '', 'OBX.18': '', 'OBX.19': '', 'OBX.20': '', 'OBX.21': '', 'OBX.22': '', 'OBX.23': '', 'OBX.24': '',
            'OBX.25': '', 'OBX.26': '', 'OBX.27': ''
        }, resultNumber-1);
    }

    const finalizedHL7 = hl7.build();
    return finalizedHL7;
}