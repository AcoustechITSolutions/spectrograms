from typing import Tuple, Optional

S3_PREFIX = 's3://'

def validate_file_ext(file_ext: str, supported_ext: Tuple[str, ...]) -> bool:
    if file_ext.endswith('"'): # Postman sends with " at the end for some reason
        file_ext = file_ext[:-1]

    if not file_ext.lower().endswith(supported_ext):
        return False
    return True

def get_file_ext(filename: str) -> Optional[str]:
    file_ext = filename.split('.')
    if not len(file_ext) == 2:
        return None
    return file_ext[1]

def get_s3_bucket(path: str) -> Optional[str]:
    pref_start = path.find(S3_PREFIX, 0)
    if (pref_start != 0):
        return None
    next_slash = path.find('/', len(S3_PREFIX), len(path))
    return path[len(S3_PREFIX):next_slash]

def get_s3_key(path: str, bucket: str) -> Optional[str]:
    pref = f'{S3_PREFIX}{bucket}/'
    pref_start = path.find(pref, 0)
    if (pref_start != 0):
        return None
    return path[len(pref):]
