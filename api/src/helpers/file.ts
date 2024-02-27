import {extname, basename} from 'path';

export const isValidFileExt = (validExtensions: Array<string>, filename: string): boolean => {
    const extension = extname(filename);
    if (validExtensions.indexOf(extension) > -1) {
        return true;
    } else {
        return false;
    }
};

export const getFileName = (fileName: string): string => {
    return basename(fileName, extname(fileName));
};

export const getFileExtName = (filename: string): string => {
    const extension = extname(filename);
    return extension.length == 0 ? extension : extension.slice(1);
};