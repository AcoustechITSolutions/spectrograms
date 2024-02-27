// import {AxiosStatic} from 'axios'

// export type PreprocessRequestDto = {
//     coughAudioPath?: string,
//     breatingAudioPath?: string,
//     speechAudioPath?: string,
//     spectreFolder: string,
//     requestId: number
// }

// type PreprocessedAudioData = {
//     samplerate: number,
//     duration: number,
//     spectre_path: string
// }

// type PreprocessResponse = {
//     cough_audio: PreprocessedAudioData,
//     breathing_audio: PreprocessedAudioData,
//     speech_audio: PreprocessedAudioData
// }

// const ML_SERVICE_TIMEOUT_MSEC = 60000

// export class MlService {
//     private axios: AxiosStatic
//     private serviceUrl: string
    
//     /**
//     * @param axios {AxiosStatic} - axios client for rest api calling.
//     * @param serviceUrl {string} - url of the service.
//      */
//     public constructor(axios: AxiosStatic, serviceUrl: string) {
//         this.axios = axios
//         this.serviceUrl = serviceUrl
//     }
//     /**
//      * Calls the preprocess dataset method on the ml service, which should
//      * generate spectrograms for passed audios and return some info about
//      * audio record such as samplerate, duration and etc. Spectrograms will be saved
//      * by the ml service, we only getting full path.
//      * @param requestDto {PreprocessRequestDto} - instance of the dto
//      */
//     public async preprocessDataset (requestDto: PreprocessRequestDto) {
//         const body = {
//             cough_audio_path: requestDto.coughAudioPath,
//             breathing_audio_path: requestDto.breatingAudioPath,
//             speech_audio_path: requestDto.speechAudioPath,
//             spectre_folder: requestDto.spectreFolder
//         }

//         try {
//             const response = this.axios.post(`${this.serviceUrl}/v1/preprocess_datasetv2/`, body, {
//                 timeout: ML_SERVICE_TIMEOUT_MSEC
//             })
//         } catch(error) {

//         }
//     }
// }