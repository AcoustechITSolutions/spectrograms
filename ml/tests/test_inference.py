import pytest
import io
from app import main
from werkzeug.datastructures import FileStorage
from flask import json
import time

@pytest.fixture
def app():
    yield main.app

@pytest.fixture
def client(app):
    main.app.config['TESTING'] = True
    yield app.test_client()

def test_inference(client):
    data = {
        'cough_audio_path': 's3://acoustery-dev/audio_data/3/38/cough.wav'
    }
   
    res = client.post('/v1/inference/',
        data=json.dumps(data),
        content_type='application/json')
   
    assert res.status_code == 200
    assert res.json['prediction'] > 0
    assert res.json['prediction'] < 1
    assert isinstance(res.json['prediction'], float)
    assert res.json['samplerate'] == 44100

    again = client.post('/v1/inference/',
        data=json.dumps(data),
        content_type='application/json')

    assert again.status_code == 200
 
    assert again.json['prediction'] > 0
    assert again.json['prediction'] < 1
    assert isinstance(again.json['prediction'], float)
    assert again.json['samplerate'] == 44100

    assert again.json['prediction'] == res.json['prediction']

    control = client.post('/v1/inference/',
        data=json.dumps(data),
        content_type='application/json')

    assert control.status_code == 200
    assert control.json['prediction'] > 0
    assert control.json['prediction'] < 1
    assert isinstance(control.json['prediction'], float)
    assert control.json['samplerate'] == 44100
    assert again.json['prediction'] == res.json['prediction'] and res.json['prediction'] == control.json['prediction']

    
    
