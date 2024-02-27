import pytest
import io
from app import main
from werkzeug.datastructures import FileStorage
from flask import json

@pytest.fixture
def app():
    yield main.app

@pytest.fixture
def client(app):
    main.app.config['TESTING'] = True
    yield app.test_client()

TEST_DATA = {'age': '25', 'gender': 'male', 'smoke': 'false', 'duration': '0', 'isforce': 'false'}

def test_diagnostic(client):
    with open('tests/test_audio/test_audio.wav', 'rb') as file:
        data = TEST_DATA
        data['cough_audio'] = (io.BytesIO(file.read()), 'file.wav')
        res = client.post('/v1/diagnostic/', data = data, 
               content_type = 'multipart/form-data'
        )
        expected_min_file_length = 651500
        assert res.status_code == 200
        assert int(res.headers['Content-Length']) >= expected_min_file_length# sanity check
        assert len(res.get_data()) >= expected_min_file_length 
        assert res.headers['Content-Type'] == 'application/pdf'

def test_diagnostic_wrong_mimetype(client):
    data = TEST_DATA
    cough_audio = FileStorage(
        stream = open('tests/test_audio/test_audio.wav', 'rb'),
        filename = 'test_audio.wav',
        content_type = 'audio/wwaw'
    )
    data['cough_audio'] = cough_audio
    
    res = client.post('/v1/diagnostic/', data = data,
            content_type = 'multipart/form-data'
    )
    expected = 'Wrong MIME type.'
    
    assert res.status_code == 400
    assert res.headers['Content-Type'] == 'application/json'
    res_json = json.loads(res.get_data(as_text = True))
    assert expected in res_json['error']

def test_diagnostic_wrong_ext(client):
    with open('tests/test_audio/wrong_audio_ext.ext.wav', 'rb') as file:
        data = TEST_DATA
        data['cough_audio'] = (io.BytesIO(file.read()), 'wrong_file.ext.wav')
        res = client.post('/v1/diagnostic/', data = data, 
             content_type = 'multipart/form-data'
        )
        expected = 'Wrong file type.'

        assert res.status_code == 400
        assert res.headers['Content-Type'] == 'application/json'
        res_json = json.loads(res.get_data(as_text = True))
        assert expected in res_json['error']
        