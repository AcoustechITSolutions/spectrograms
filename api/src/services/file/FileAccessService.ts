import {Readable} from 'stream';

export type UploadFile = {
    data: Buffer,
    path: string
}

/**
 * Responsible for file manipulation.
 */
export interface FileAccessService {
    /**
     * Saves file on some service(s3, ftp, file system) and returns full path. 
     * This path should be saved and used in other methods.
     * ATTENTION!! Cannot work in parallel due to sftp restrictions.
     * @param path {string} - path to file with all folders included e.g. first_folder/user/file.ext
     * @param data {Buffer} - buffer with file data
     */
    saveFile(path: string, data: Buffer): Promise<string> // TODO: refactor to save files in folder, because sftp cannot work in parallel.
    /**
     * Returns file content as node js readable stream.
     * @param fullPath {string} - full path of a file which was returned from the saveFile() method.
     */
    getFileAsStream(fullPath: string): Promise<Readable>
    /**
     * Deletes file by its full path.
     * @param fullPath {string} - full path of a file which was returned from the saveFile() method.
    */
    deleteFile(fullPath: string): Promise<void>
    /**
     * Recursively deletes all files in a directory.
     * @param dirFullPath {string}: full path to the directory.
     */
    deleteDirectory(dirFullPath: string): Promise<void>
}
