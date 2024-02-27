from app.services.file.file_access import FileAccessService
from app.utils import validate_file_ext, get_file_ext, get_s3_bucket, get_s3_key
import io

class S3FileAccessImpl(FileAccessService):

    def __init__(self, s3, bucket):
        self.__s3 = s3
        self.__bucket = bucket

    def upload_file(self, file_path: str, data: io.BytesIO) -> str:
        self.__s3.put_object(
            Bucket=self.__bucket,
            Key=file_path,
            Body=data
        ) 
        return f's3://{self.__bucket}/{file_path}'

    def download_file(self, full_path: str, data: io.BytesIO) -> None:
        bucket = get_s3_bucket(full_path)
        key = get_s3_key(full_path, bucket)
        if bucket is None or key is None:
            raise Exception('invalid_cough_path')
        self.__s3.download_fileobj(bucket, key, data)
