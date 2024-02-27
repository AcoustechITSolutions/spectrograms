import {S3} from 'aws-sdk';
import {FileAccessService} from './FileAccessService';
import {Readable} from 'stream';
import {dirname} from 'path';

export class S3FileAccessServiceImpl implements FileAccessService {
    
    public constructor(private s3: S3, private bucket: string, private spectreFolder: string) {}

    public async saveFile(path: string, data: Buffer): Promise<string> {
        await this.s3.putObject({
            Body: data,
            Bucket: this.bucket,
            Key: path,
        }).promise();
        const fullPath = `s3://${this.bucket}/${path}`;
        return fullPath;
    }

    public async getFileAsStream(fullPath: string): Promise<Readable> {
        return this.s3.getObject({
            Bucket: this.bucket,
            Key: this.getS3PathKey(fullPath),
        })
            .createReadStream();
    }

    public async deleteFile(fullPath: string): Promise<void> {
        await this.s3.deleteObject({
            Bucket: this.bucket,
            Key: this.getS3PathKey(fullPath)
        }).promise();
    }

    public async deleteDirectory(dirFullPath: string): Promise<void> {
        await this.emptyS3Directory(this.bucket, dirname(this.getS3PathKey(dirFullPath)));
    }

    private getS3PathKey (path: string): string | null {
        const prefix = `s3://${this.bucket}/`;
        const index = path.indexOf(prefix);
        if (index == -1) {
            return null;
        }
        return path.substring(index + prefix.length);
    }
    
    private async emptyS3Directory (bucket: string, dir: string) {
        const listParams = {
            Bucket: bucket,
            Prefix: dir,
        };
    
        const listedObjects = await this.s3.listObjectsV2(listParams).promise();
        if (listedObjects.Contents.length === 0) return;
    
        const objects = new Array<S3.ObjectIdentifier>();
    
        listedObjects.Contents.forEach( (object) => {
            const id: S3.ObjectIdentifier = {
                Key: object.Key,
            };
            objects.push(id);
        });
    
        await this.s3.deleteObjects({
            Bucket: bucket,
            Delete: {
                Objects: objects,
            },
        }).promise();
    
        if (listedObjects.IsTruncated) await this.emptyS3Directory(bucket, dir);
    }
}
    