from app.services.file.file_access import FileAccessService
from paramiko import SFTPClient
import io
from os import path

class SftpFileAccessImpl(FileAccessService):
    def __init__(self, sftp: SFTPClient, upload_folder: str):
        self.__sftp = sftp
        self.__upload_folder = upload_folder
    
    def upload_file(self, file_path: str, data: io.BytesIO) -> str:
        upload_path = f'{self.__upload_folder}/{file_path}'
        directory = path.dirname(upload_path)
        try:
            self.__sftp.mkdir(directory)
        except IOError as e:
            pass # Assume that dir is already exists
        self.__sftp.putfo(data, upload_path)
        return f'sftp://{upload_path}'

    def download_file(self, full_path: str, data: io.BytesIO) -> None:
        prefix = 'sftp://'
        prefix_start = full_path.find(prefix, 0)
        if (prefix_start != 0):
            raise Exception('invalid_path')
        folder_path = full_path[len(prefix):]
        self.__sftp.getfo(folder_path, data)
