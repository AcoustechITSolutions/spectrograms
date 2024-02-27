import pytest
from app.utils import *

def test_valid_file_ext():
    filename = 'test.wav'
    res = get_file_ext(filename)
    assert res is not None
    assert res == 'wav'

def test_invalid_file_ext():
    filename = 'test.t.wav'
    res = get_file_ext(filename)
    assert res is None

def test_valid_validation():
    ext = 'wav'
    res = validate_file_ext(ext, 'wav')
    assert res == True

def test_invalid_validation():
    ext = 'ogg'
    res = validate_file_ext(ext, 'wav')
    assert res == False

def test_validate_postman_filename():
    ext = 'wav"'
    res = validate_file_ext(ext, 'wav')
    assert res == True
