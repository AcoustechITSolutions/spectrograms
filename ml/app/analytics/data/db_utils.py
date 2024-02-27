#!/usr/bin/env
import os
import json

import boto3
from botocore.exceptions import ClientError
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


S3_PREFIX = 's3://'
TMP_WORK_DIR = '/tmp/acoustery'
USER = 'ml'
PASSWORD = '=7RtQIO-^zys'
SSH_USERNAME = 'tunnel'
SSH_IP = '18.158.151.87'
SSH_PORT = 22
STORAGE_BIND_ADDRESS = 'acousterydb.c8qssl9yutix.eu-central-1.rds.amazonaws.com'
AWS_ACCESS_KEY = 'AKIA2AXFKPT2CK6J4JSR'
AWS_SECRET_ACCESS_KEY = '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'
BUCKET_NAME = 'acoustery'


def get_engine_for_port(port):
    return create_engine('postgresql://{user}:{password}@{host}:{port}/{db}'.format(
        user=USER,
        password=PASSWORD,
        host='127.0.0.1',
        port=port,
        db='acoustery_db'
    ), poolclass=NullPool)


def ssh_forwarder(conf):
    return SSHTunnelForwarder(
                (SSH_IP, SSH_PORT),
                ssh_username=SSH_USERNAME,
                ssh_pkey=conf.permission_key,
                ssh_private_key_password='',
                remote_bind_address=(STORAGE_BIND_ADDRESS, 5432)
    )


def load_db_metadata(config):
    """
    Loads train and eval metadata from database. Which type of data to load specified inf config file
    :param config: add dict or pyhocon config object
    :return list with metadata
    """
    request = f"""
            with mytable as (
                select 
                    audio.audio_path as path,
                    audio.samplerate as samplerate,
                    covid19.symptomatic_type as symptomatic,
                    types.audio_type,
                    audio.is_representative,
                    audio.is_representative_scientist,
                    users.login as data_source,
                    req.doctor_status_id,
                    req.marking_status_id,
                    cough_char.commentary,
                    audio.is_marked,
                    req.user_id as uid,
                    req.date_created as date
                from dataset_audio_info as audio
                join dataset_cough_characteristics as cough_char
                    on audio.request_id = cough_char.request_id
                join dataset_request as req
                    on audio.request_id = req.id
                join dataset_patient_diseases as patient
                    on req.id = patient.request_id
                join covid19_symptomatic_types as covid19
                    on covid19.id = patient.covid19_symptomatic_type_id
                join dataset_audio_types as types
                    on types.id = audio.audio_type_id
                join users
                    on users.id = req.user_id
            ), 
            episodes as (
                select 
                    audio.audio_path as audio_path,
                    array_agg(episodes."start" order by episodes."start") as "start",
                    array_agg(episodes."end" order by episodes."end") as "end"
                from dataset_audio_info audio
                join dataset_audio_espisodes episodes
                    on episodes.audio_info_id = audio.id
                group by audio.audio_path
            )

            select 
                path,
                samplerate,
                "start",
                "end",
                symptomatic
            from mytable
            left join episodes on mytable.path = episodes.audio_path
            where mytable.audio_type = 'cough'
                    and mytable.data_source not in {config['now_downloadable']}
                    and mytable.is_representative_scientist = 'true'
                    ;
            """
    request_ids = """
        with counttable as (select request_id,count(audio_type_id) from (
        select 
            dataset_audio_info.request_id,
            audio_type_id,
            is_representative,
            is_marked,
            dataset_patient_diseases.covid19_symptomatic_type_id,
            dataset_patient_diseases.disease_type_id,
            dataset_request.marking_status_id,
            dataset_cough_characteristics.commentary 
        from dataset_audio_info
    join dataset_patient_diseases
        on dataset_audio_info.request_id = dataset_patient_diseases.request_id
    join dataset_cough_characteristics
        on dataset_audio_info.request_id = dataset_cough_characteristics.request_id
    join dataset_request
        on dataset_audio_info.request_id = dataset_request.id) as mytable
    where (audio_type_id = 1 and mytable.is_marked = 'true' and mytable.is_representative = 'true') 
        or (audio_type_id = 2 and ((mytable.is_marked = 'true' and mytable.is_representative = 'true') or mytable.commentary LIKE '%#is_repr%')) 
    group by request_id)
    select request_id from counttable
    where count = 2
    order by request_id
    
    """

    table = f"""
        with mytable as (
            select 
                audio.request_id as reqq_id,
                audio.audio_path as path,
                audio.samplerate as samplerate,
                covid19.symptomatic_type as symptomatic,
                types.audio_type,
                audio.is_representative,
                users.login as data_source,
                req.doctor_status_id,
                req.marking_status_id,
                cough_char.commentary,
                audio.is_marked,
                req.user_id as uid,
                req.date_created as date
            from dataset_audio_info as audio
            join dataset_cough_characteristics as cough_char
                on audio.request_id = cough_char.request_id
            join dataset_request as req
                on audio.request_id = req.id
            join dataset_patient_diseases as patient
                on req.id = patient.request_id
            join covid19_symptomatic_types as covid19
                on covid19.id = patient.covid19_symptomatic_type_id
            join dataset_audio_types as types
                on types.id = audio.audio_type_id
            join users
                on users.id = req.user_id
        ), 
        episodes as (
            select 
                audio.audio_path as audio_path,
                array_agg(episodes."start" order by episodes."start") as "start",
                array_agg(episodes."end" order by episodes."end") as "end"
            from dataset_audio_info audio
            join dataset_audio_espisodes episodes
                on episodes.audio_info_id = audio.id
            group by audio.audio_path
        )

        select 
            reqq_id,
            date,
            path,
            samplerate,
            "start",
            "end",
            symptomatic
        from mytable
        left join episodes on mytable.path = episodes.audio_path
        where mytable.uid > 40
            
            and mytable.audio_type=
        """

    cough_and_breath_sql_request = f"""
        create view cough as ({table}'cough');
        create view breath as ({table}'breathing');  
        create view request_ids as ({request_ids});
        
                select
            cough.path,
            breath.path,
            cough.samplerate,
            cough."start",
            cough."end",
            breath."start",
            breath."end",
            cough.symptomatic,
            cough.reqq_id
        from 
            cough,
            breath 
        where 
            cough.date = breath.date
            and exists (select * from request_ids where request_id = cough.reqq_id)
        ;
      
    """

    with ssh_forwarder(config) as tunnel:
        tunnel.start()
        engine = get_engine_for_port(tunnel.local_bind_port)
        session = sessionmaker(bind=engine)()
        req = cough_and_breath_sql_request if config.audio_type == 'cough_and_breath' \
            else request
        db_info = session.execute(req).fetchall()
        session.close()
        return [tuple(item) for item in db_info if item[-1] != 'likely_covid19']


def get_s3_key(path: str, bucket: str):
    pref = f'{S3_PREFIX}{bucket}/'
    pref_start = path.find(pref, 0)
    if pref_start != 0:
        return None
    return path[len(pref):]


def create_bucket():
    aws_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='eu-central-1'
    )

    s3 = aws_session.resource('s3')

    return s3.Bucket(BUCKET_NAME)


def download_audio_files(config, metadata):
    """
    Download audio files from S3 storage
    :param config: add dict or pyhocon config object
    :param metadata: list with audio file names in S3 storage and additional data
    :return dict with audio names as key and file metadata as value
    """

    bucket = create_bucket()

    out_desc, dictionary = {}, {0: "cough", 1: "breathing"}
    # make temp directory for swamp files os use existing
    if 'cache_data' in config:
        work_dir = config.cache_data
        if os.path.isfile(f'{work_dir}/metadata.json'):
            with open(f'{work_dir}/metadata.json') as json_file:
                out_desc = json.load(json_file)
    else:
        work_dir = TMP_WORK_DIR

    if config.audio_type == 'cough_and_breath':
        for type_val in dictionary.values():
            if not os.path.isdir(f'{work_dir}/{type_val}'):
                os.makedirs(f'{work_dir}/{type_val}')
    else:
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

    req_desc = {}

    def find_or_load(a_type, f_dir, a_name, sr, start, end, sympt, double=False):
        work_directory = f'{work_dir}/{a_type}' if double else f'{work_dir}'
        f_num = len(os.listdir(work_directory))
        f_name = f'{f_dir}/db_file_{f_num}.wav'
        s3_key = get_s3_key(a_name, BUCKET_NAME)
        try:
            info = {'audio_type': a_type,
                    'sr': sr,
                    'start': start,
                    'end': end,
                    'symptomatic': sympt,
                    }
            if s3_key not in out_desc or\
                    not info.items() <= out_desc[s3_key].items():
                with open(f_name, 'wb') as f:
                    bucket.download_fileobj(s3_key, f)
                out_desc[s3_key] = info
                out_desc[s3_key]['file_path'] = f_name
            req_desc[s3_key] = info
            req_desc[s3_key]['file_path'] = out_desc[s3_key]['file_path']
        except ClientError:
            print(f'Audio file {s3_key} was not found in AWS bucket.')
            os.remove(f_name)

    for idx, db_item in enumerate(metadata):
        if idx % 100 == 0 and idx != 0:
            print(f'Loaded {idx} audio files.')
        if config.audio_type == 'cough_and_breath':
            for key, val in dictionary.items():
                # download audio track from S3 only in case there's no local version
                if val == 'cough':
                    find_or_load(val, f'{work_dir}/{val}', db_item[key], db_item[2], db_item[3], db_item[4], db_item[7], double=True)
                elif val == 'breathing':
                    find_or_load(val, f'{work_dir}/{val}', db_item[key], db_item[2], db_item[5], db_item[6], db_item[7], double=True)
        else:
            find_or_load(config.audio_type, f'{work_dir}', *db_item)

    # save dataset metadata in json file
    with open(f'{work_dir}/metadata.json', 'w') as json_file:
        json.dump(out_desc, json_file)

    return req_desc
