from abc import ABC, abstractclassmethod
import io

class FileAccessService(ABC):
    """
    Should upload file by its path. 
    Returns full path with protocol name.
    """
    @abstractclassmethod
    def upload_file(self, file_path: str, data: io.BytesIO) -> str:
        pass

    """
    Should download file by its full path to file object or BytesIO instance.
    """
    @abstractclassmethod
    def download_file(self, full_path: str, data: io.BytesIO) -> None:
        pass