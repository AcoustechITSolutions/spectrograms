import {getErrorMessage, HttpErrors, HttpStatusCodes} from '../helpers/status';
import {Request, Response} from 'express';
import {UploadedFile} from 'express-fileupload';
import {getFileExtName, isValidFileExt} from '../helpers/file';
import FormData from 'form-data';
import config from '../config/config';
import axios from 'axios';

const validExtensions = ['.ogg'];

export class ConverterController {
    public async convert (req: Request, res: Response) {
        let file: UploadedFile;
        try {
            file = req.files.file as UploadedFile;
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INCORRECT_BODY);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }
 
        if(!isValidFileExt(validExtensions, file.name)) {
            const errorMessage = getErrorMessage(HttpErrors.FILE_FORMAT_ERROR);
            return res.status(HttpStatusCodes.BAD_REQUEST).send(errorMessage);
        }

        const extension = getFileExtName(file.name);
        const form = new FormData();
        form.append(
            'file',
            file.data,
            {
                contentType: `audio/${extension}`,
                filename: `converter_file.${extension}`
            }
        );

        let response;
        res.setHeader('Content-Type', 'audio/wav');
        try {
            response = await axios.post(`${config.mlServiceURL}/v1.1/convert/`, form, {
                headers: {...form.getHeaders()},
                responseType: 'stream'
            });
        } catch(error) {
            console.error(error);
            const errorMessage = getErrorMessage(HttpErrors.INTERNAL_ERROR);
            return res.status(HttpStatusCodes.ERROR).send(errorMessage);
        }
        return response.data.pipe(res);
    }
}