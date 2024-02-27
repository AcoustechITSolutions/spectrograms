import {FileAccessService} from './FileAccessService';
import SftpClient from 'ssh2-sftp-client';
import {Readable} from 'stream';
import {dirname} from 'path';

export class SftpFileAccessServiceImpl implements FileAccessService {

    public constructor(private sftp: SftpClient, private uploadFolder: string, 
        private spectreFolder: string) {}

    public async saveFile(path: string, data: Buffer): Promise<string> {
        const filePath = `${this.uploadFolder}/${path}`;
        const fileDir = await dirname(filePath);
    
        const isPathExists = await this.sftp.exists(fileDir);
        if (!isPathExists) {
            await this.sftp.mkdir(fileDir, true);
        }

        await this.sftp.put(data, filePath);
        return `sftp://${this.uploadFolder}/${path}`;
    }

    public async getFileAsStream(fullPath: string): Promise<Readable> {
        const path = this.getPathFromFullPath(fullPath);
        const res = await this.sftp.get(path) as Buffer;
        console.log('got this res');
        console.log(res);
        const readable = Readable.from(res);
        return readable;
    }

    private getPathFromFullPath(fullPath: string): string | null {
        const prefix = 'sftp://';
        const index = fullPath.indexOf(prefix);
        if (index == -1)
            return null;
        return fullPath.substring(index + prefix.length);
    }

    public async deleteFile(fullPath: string): Promise<void> {
        await this.sftp.delete(this.getPathFromFullPath(fullPath));
    }

    public async deleteDirectory(fullPath: string): Promise<void> {
        const path = this.getPathFromFullPath(fullPath);
        const dirPath = dirname(path);
        await this.sftp.rmdir(dirPath, true);
    }
}
