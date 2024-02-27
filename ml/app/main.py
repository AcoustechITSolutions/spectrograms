from flask import Flask, jsonify, Response, request, send_file, make_response, json, jsonify
from distutils import util
import os, sys, shutil, uuid
import io
import soundfile
import boto3
import librosa
import onnxruntime
import matplotlib
import traceback
from app.utils import validate_file_ext, get_file_ext, get_s3_bucket, get_s3_key
from app.services.file.file_access import FileAccessService
from app.services.file.s3_file_access import S3FileAccessImpl
from app.services.file.sftp_file_access import SftpFileAccessImpl
import paramiko
import numpy

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), 'app/analytics/'))

from tools.full_api.inference import detect, make_records_for_detector, make_input, noise_classification, \
    covid_classification, waveform_noise_reduction, freq_THRESHOLD, lower_noise_THRESHOLD
from tools.speaker_ID.main import make_embed
from app.audio_processing import generate_spectrogram

MODEL_SHARE_FOLDER = os.environ['ML_MODELS_SHARE_DIR']
CLASSIFICATION_MODEL_NAME = 'batch_norm_full_89.onnx'
CLASSIFICATION_MODEL_PATH = f'{MODEL_SHARE_FOLDER}/{CLASSIFICATION_MODEL_NAME}'
DETECTOR_MODEL_NAME = 'acoustery_detector.onnx'
DETECTOR_MODEL_PATH = f'{MODEL_SHARE_FOLDER}/{DETECTOR_MODEL_NAME}'
NOISER_MODEL_NAME = 'embedder.onnx'
NOISER_MODEL_PATH = f'{MODEL_SHARE_FOLDER}/{NOISER_MODEL_NAME}'

ML_TEMP_DIR = os.environ['ML_TEMP_DIR']
TEMP_AUDIO_DIR = os.environ['TEMP_AUDIO_DIR']

classification_model = None
detector_model = None
noiser_model = None

if os.environ['NODE_ENV'] != 'production':
    aws_endpoint = os.environ['LOCALSTACK_ENDPOINT']
else:
    aws_endpoint = None

FILE_ACCESS_PROTOCOL = os.environ['FILE_ACCESS_PROTOCOL']
if FILE_ACCESS_PROTOCOL == 's3':
    if 'S3_USER_ID' in os.environ and 'S3_USER_SECRET' in os.environ:
        aws_access_key_id = os.environ['S3_USER_ID']
        aws_secret_access_key = os.environ['S3_USER_SECRET']
    else:
        aws_access_key_id = None
        aws_secret_access_key = None

    s3 = boto3.client('s3',
        endpoint_url = aws_endpoint,
        region_name = 'eu-central-1',
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    file_access: FileAccessService = S3FileAccessImpl(s3, os.environ['S3_BUCKET'])
elif FILE_ACCESS_PROTOCOL == 'sftp':
    hostname = os.environ['SFTP_HOST']
    port = int(os.environ['SFTP_PORT'])
    t = paramiko.Transport((hostname, port))
    t.start_client()
    t.auth_password(
        username = os.environ['SFTP_USERNAME'],
        password = os.environ['SFTP_PASSWORD']
    )
    sftp = paramiko.SFTPClient.from_transport(t)
    file_access: FileAccessService = SftpFileAccessImpl(sftp, os.environ['SFTP_UPLOAD_FOLDER'])
else:
    raise Exception('no file access protocol provided')

def preload():
    download_actual_model()
    init_model()

def init_model():
    print('Loading classification model', flush = True)
    global classification_model 
    classification_model = onnxruntime.InferenceSession(CLASSIFICATION_MODEL_PATH)
    print('Classification model loaded', flush = True)
    
    print('Loading detector model', flush = True)
    global detector_model
    detector_model = onnxruntime.InferenceSession(DETECTOR_MODEL_PATH)
    print('Detector model loaded', flush = True)

    print('Loading noise classification model', flush = True)
    global noiser_model
    noiser_model = onnxruntime.InferenceSession(NOISER_MODEL_PATH)
    print('Noise classification model loaded', flush = True)

def download_actual_model():
    print('check for actual models', flush = True)
    if not os.path.exists(CLASSIFICATION_MODEL_PATH):
        raise Exception("No classification model") 
    
    if not os.path.exists(DETECTOR_MODEL_PATH):
        raise Exception("No detector model")

    if not os.path.exists(NOISER_MODEL_PATH):
        raise Exception("No noise classification model")

app = Flask(__name__)
app.before_first_request(preload)
matplotlib.use('agg')

AUDIO_DIR = os.environ['AUDIO_SHARE_DIR']
SPECTRE_DIR = os.environ['DATASET_SPECTRE_FOLDER']
SUPPORTED_EXTENSIONS = ('wav')
SUPPORTED_MIME_TYPES = {'audio/wav', 'audio/x-wav', 'audio/vnd.wave', 'audio/wave'}

@app.errorhandler(FileNotFoundError)
def handle_no_file(e: FileNotFoundError) -> Response:
    return Response(status = 500)

@app.route('/test', methods=['GET'])
def handle_test() -> str:
    return 'Working'

@app.errorhandler(Exception)
def handle_excp(e: Exception) -> Response:
    print(e, flush = True)
    traceback.print_exc()
    return Response(status = 500)

@app.route('/v1/detection/cough/', methods=['POST'])
def handle_cough_detection() -> Response:
    cough_audio = request.files['cough_audio']
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{cough_audio.filename}'
    cough_audio.save(file_path)

    global detector_model
    track, sr = librosa.load(file_path, sr = 44100)
    inference_result = detect(detector_model, make_records_for_detector(track, sr))
    print(f'Number of coughs detected in {file_path} is {inference_result}', flush = True)
    audio_duration = track.shape[0] / sr
    cough_frequency = inference_result / audio_duration
    print(f'Cough frequency: {cough_frequency}', flush = True)
    os.remove(file_path)
    return make_response({
        'is_detected' : bool(inference_result > 0),
        'is_enough' : bool(cough_frequency > 0.5)
    })

@app.route('/v1/noisy/cough/', methods=['POST'])
def handle_noise_classification() -> Response:
    cough_audio = request.files['cough_audio']
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{cough_audio.filename}'
    cough_audio.save(file_path)

    global noiser_model
    track, sr = librosa.load(file_path, sr = 44100)
    track_denoised = waveform_noise_reduction(track, sr)
    if not numpy.isfinite(track).all():
        track = numpy.nan_to_num(track, nan=0, posinf=1.0, neginf=-1.0)
    if not numpy.isfinite(track_denoised).all():
        track_denoised = numpy.nan_to_num(track_denoised, nan=0, posinf=1.0, neginf=-1.0)

    inference_result = noise_classification(noiser_model, track, track_denoised, sr)
    print(f'Probability of not being noisy for {file_path} is {inference_result}', flush = True)
    os.remove(file_path)
    return make_response({
        'is_noisy' : bool(inference_result < lower_noise_THRESHOLD)
    })

@app.route('/v1/validation/cough/', methods=['POST'])
def handle_cough_validation() -> Response:
    cough_audio = request.files['cough_audio']
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{cough_audio.filename}'
    cough_audio.save(file_path)

    global detector_model
    global noiser_model
    track, sr = librosa.load(file_path, sr = 44100)
    track_denoised = waveform_noise_reduction(track, sr)
    if not numpy.isfinite(track).all():
        track = numpy.nan_to_num(track, nan=0, posinf=1.0, neginf=-1.0)
    if not numpy.isfinite(track_denoised).all():
        track_denoised = numpy.nan_to_num(track_denoised, nan=0, posinf=1.0, neginf=-1.0)

    detection_result = detect(detector_model, make_records_for_detector(track, sr))
    print(f'Number of coughs detected in {file_path} is {detection_result}', flush = True)
    audio_duration = track.shape[0] / sr
    cough_frequency = detection_result / audio_duration
    print(f'Cough frequency: {cough_frequency}', flush = True)

    noiser_result = noise_classification(noiser_model, track, track_denoised, sr)
    print(f'Probability of not being noisy for {file_path} is {noiser_result}', flush = True)
    os.remove(file_path)
    return make_response({
        'is_cough' : bool(detection_result > 0),
        'is_clear' : bool(noiser_result > lower_noise_THRESHOLD),
        'is_enough' : bool(cough_frequency > 0.5)
    })

@app.route('/v1.2/voice/embedding/', methods=['POST'])
def handle_voice_embedding() -> Response:
    speech_audio = request.files['speech_audio']
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{speech_audio.filename}'
    speech_audio.save(file_path)

    global noiser_model
    track, sr = librosa.load(file_path, sr = 44100)
    embedding = make_embed(noiser_model, track, sr)
    embedding_string = ','.join(map(str, embedding))
    print(f'Voice embedding for {file_path} is done', flush = True)
    os.remove(file_path)
    return make_response({
        'voice_embedding' : embedding_string
    })

@app.route('/v1/inference/', methods=['POST']) 
def handle_inference() -> Response:
    cough_audio_path = request.json['cough_audio_path']
    
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{os.path.basename(cough_audio_path)}'
    try:
        with open(file_path, 'wb') as f:
            file_access.download_file(cough_audio_path, f)
    except Exception as e:
        print(e)
        print(f'invalid path {cough_audio_path}', flush = True)
        return Response(status = 400)

    global classification_model
    track, sr = librosa.load(file_path, sr = 44100)
    audio_duration = track.shape[0] / sr

    inference_result = covid_classification(classification_model, make_input(track))

    os.remove(file_path)
    print(f'Covid probability for {file_path} is {inference_result}', flush = True)
    return  make_response({
        'prediction': inference_result,
        'samplerate': sr,
        'audio_duration': audio_duration
    })

@app.route('/v1.1/public/inference/', methods=['POST']) 
def handle_public_inference() -> Response:
    if "cough_audio" not in request.files:
        print(f'cough_audio file required', flush = True)
        return Response(status = 400)
    cough_audio = request.files["cough_audio"]    
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{os.path.basename(cough_audio.filename)}'
    cough_audio.save(file_path)

    global classification_model
    track, sr = librosa.load(file_path, sr = 44100)
    audio_duration = track.shape[0] / sr

    inference_result = covid_classification(classification_model, make_input(track))
    
    os.remove(file_path)
    print(f'Covid probability for {file_path} is {inference_result}', flush = True)
    return  make_response({
        'prediction': inference_result,
        'samplerate': sr,
        'audio_duration': audio_duration
    })

@app.route('/v1/preprocess_datasetv2/', methods = ['POST'])
def handle_preprocess_v2() -> Response:
    cough_audio_path = None
    breathing_audio_path = None
    speech_audio_path = None
    if 'cough_audio_path' in request.json:
        cough_audio_path = request.json['cough_audio_path']
    if 'breathing_audio_path' in request.json:
        breathing_audio_path = request.json['breathing_audio_path']
    if 'speech_audio_path' in request.json:
        speech_audio_path = request.json['speech_audio_path']
    spectre_folder = request.json['spectre_folder']

    print(f'preprocess for {cough_audio_path}', flush = True)

    if breathing_audio_path is not None:
        breathing_name = f'{TEMP_AUDIO_DIR}/{uuid.uuid4()}_{os.path.basename(breathing_audio_path)}'
        with open(breathing_name, 'wb') as breathing_audio:
            file_access.download_file(breathing_audio_path, breathing_audio)
    if cough_audio_path is not None:
        cough_name = f'{TEMP_AUDIO_DIR}/{uuid.uuid4()}_{os.path.basename(cough_audio_path)}'
        with open(cough_name, 'wb') as cough_audio:
            file_access.download_file(cough_audio_path, cough_audio) 
    if speech_audio_path is not None:
        speech_name = f'{TEMP_AUDIO_DIR}/{uuid.uuid4()}_{os.path.basename(speech_audio_path)}'
        with open(speech_name, 'wb') as speech_audio:
            file_access.download_file(speech_audio_path, speech_audio)
    
    cough_samplerate, cough_audio_spectre_path, cough_audio_duration, cough_audio_spectre_full_path = \
        None, None, None, None
    if cough_audio_path is not None:
        cough_audio_clean, cough_samplerate = librosa.load(cough_name, sr=None)
        cough_audio_duration = cough_audio_clean.shape[0] / cough_samplerate
        cough_audio_spectre = generate_spectrogram(cough_audio_clean, cough_samplerate) 
        cough_audio_spectre_path = f'{spectre_folder}/cough_audio_spectre.png'
        cough_audio_spectre_full_path = file_access.upload_file(cough_audio_spectre_path, cough_audio_spectre)
        os.remove(cough_name)

    breathing_audio_duration, breathing_samplerate, breathing_audio_spectre_path, breathing_audio_spectre_full_path = \
        None, None, None, None
    if breathing_audio_path is not None:
        breathing_audio_clean, breathing_samplerate = librosa.load(breathing_name, sr=None)
        breathing_audio_duration = breathing_audio_clean.shape[0] / breathing_samplerate
        breathing_audio_spectre = generate_spectrogram(breathing_audio_clean, breathing_samplerate)
        breathing_audio_spectre_path = f'{spectre_folder}/breathing_audio_spectre.png'
        breathing_audio_spectre_full_path = file_access.upload_file(breathing_audio_spectre_path, breathing_audio_spectre)
        os.remove(breathing_name)

    speech_audio_duration, speech_samplerate, speech_audio_spectre_path, speech_audio_spectre_full_path = \
        None, None, None, None
    if speech_audio_path is not None:
        speech_audio_clean, speech_samplerate = librosa.load(speech_name, sr=None)
        speech_audio_duration = speech_audio_clean.shape[0] / speech_samplerate
        speech_audio_spectre = generate_spectrogram(speech_audio_clean, speech_samplerate)
        speech_audio_spectre_path = f'{spectre_folder}/speech_audio_spectre.png'
        speech_audio_spectre_full_path = file_access.upload_file(speech_audio_spectre_path, speech_audio_spectre)
        os.remove(speech_name)

    print(f'returning answer for {cough_audio_path}', flush = True)

    return make_response({
        'cough_audio': {
            'samplerate': cough_samplerate,
            'duration': cough_audio_duration,
            'spectre_path': cough_audio_spectre_full_path
        },
        'breathing_audio': {
            'samplerate': breathing_samplerate,
            'duration': breathing_audio_duration,
            'spectre_path': breathing_audio_spectre_full_path
        },
        'speech_audio': {
            'samplerate': speech_samplerate,
            'duration': speech_audio_duration,
            'spectre_path': speech_audio_spectre_full_path
        }
    })

@app.route('/v1/preprocess_audio/', methods = ['POST'])
def handle_preprocess_audio() -> Response:
    audio_path = request.json['audio_path']
    audio_type = request.json['audio_type']
    spectre_folder = request.json['spectre_folder']
    print(f'preprocess audio for {audio_path}', flush = True)

    bucket = get_s3_bucket(audio_path)
    
    audio_bytes = io.BytesIO()
    file_access.download_file(audio_path, audio_bytes)

    audio_bytes.seek(0)
    audio_clean, audio_samplerate = soundfile.read(audio_bytes)
    audio_duration = audio_clean.shape[0] / audio_samplerate
    audio_spectre = generate_spectrogram(audio_clean, audio_samplerate)
    spectrogram_path = f'{spectre_folder}/{audio_type}_audio_spectre.png'
    spectrogram_full_path = file_access.upload_file(spectrogram_path, audio_spectre)

    print(f'returning successfull from preprocess audio for {audio_path}')
    return make_response({
        'spectrogram_path': spectrogram_full_path,
        'samplerate': audio_samplerate,
        'duration': audio_duration
    })

@app.route('/v1.1/convert/', methods = ['POST'])
def handle_convert_audio() -> Response:
    print('converting', flush = True)
    if 'file' not in request.files:
        print('no file for convert', flush = True)
        return Response(status = 400)
    file = request.files["file"]
    file_path = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{file.filename}'
    file.save(file_path)
    y, sr = librosa.load(file_path, sr = None)
    output_file = io.BytesIO()
    soundfile.write(output_file, y, sr, format = 'WAV')
    os.remove(file_path)
    output_file.seek(0)
    return send_file(
        output_file,
        as_attachment = True,
        attachment_filename = 'audio.wav',
        mimetype = 'audio/wav'
    )

@app.route('/v1.2/spectrogram/', methods = ['POST'])
def get_spectrogram() -> Response:
    cough_audio_path = request.json['cough_audio_path']
    spectre_folder = request.json['spectre_folder']

    print(f'generating spectrogram for {cough_audio_path}', flush = True)

    cough_name = f'{ML_TEMP_DIR}/{uuid.uuid4()}_{os.path.basename(cough_audio_path)}'
    with open(cough_name, 'wb') as cough_audio:
        file_access.download_file(cough_audio_path, cough_audio) 
    
    cough_audio_clean, cough_samplerate = librosa.load(cough_name, sr=None)
    cough_audio_spectre = generate_spectrogram(cough_audio_clean, cough_samplerate) 
    cough_audio_spectre_path = f'{spectre_folder}/cough_audio_spectre.png'
    cough_audio_spectre_full_path = file_access.upload_file(cough_audio_spectre_path, cough_audio_spectre)
    os.remove(cough_name)

    print(f'generated spectrogram: {cough_audio_spectre_full_path}', flush = True)

    return make_response({
        'spectre_path': cough_audio_spectre_full_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
