import sys
import pdfkit
import jinja2
import os
import uuid
from typing import Optional
sys.path.append(os.path.join(os.path.abspath(os.getcwd()), 'app/analytics/'))

TEMPLATE_FILE = 'template.v2.html'
PDF_DIR = os.environ['PDF_SHARE_DIR']
CURR_DIR = os.path.abspath(os.getcwd())
TEMPLATE_DIR = os.path.join(CURR_DIR, 'app/template')

def str2pdf(content: str, filename: str) -> None:
        options = {
            'enable-local-file-access':'',
            'margin-top': '0',
            'margin-bottom': '0',
            'margin-left': '0',
            'margin-right': '0',
            'disable-smart-shrinking': '',
            'page-size': 'A4',
            'encoding': 'utf-8'
        }
        pdfkit.from_string(content, filename, options = options)

def create_pdf(parameters: dict) -> Optional[str]:
    loader = jinja2.FileSystemLoader(searchpath = TEMPLATE_DIR, encoding="utf-8")
    template_env = jinja2.Environment(loader = loader)
    template = template_env.get_template(TEMPLATE_FILE)
    output = render_template(template, parameters, TEMPLATE_DIR)
    folder_path = f"{PDF_DIR}/{parameters['user_id']}/{parameters['request_id']}/"
    os.makedirs(folder_path, mode=0o777, exist_ok=True)
    pdf_file_path = os.path.join(folder_path, "result.pdf")
    print(pdf_file_path)
    try:
        str2pdf(str(output), pdf_file_path)
        return pdf_file_path
    except Exception as e:
        print(e, flush = True)
        return None

def bool2str(param: bool) -> str:
    if param:
        return 'Да'
    return 'Нет'

def intensity2str(intensity: str) -> str:
    if intensity == 'paroxysmal':
        return 'Приступообразный'
    elif intensity == 'paroxysmal_hacking':
        return 'Приступообразный, надсадный'
    else:
        return 'Не приступообразный'

def productivity2str(productivity: str) -> str:
    if productivity == 'productive':
        return 'Продуктивный'
    elif productivity == 'wet_productive_small':
        return 'Мокрый/малопродуктивный'
    elif productivity == 'dry_productive_small':
        return 'Сухой/малопродуктивный'
    else:
        return 'Не продуктивный'

def diagnosis2str(diagnosis: str) -> str:
    if diagnosis == 'covid_19':
        return 'COVID-19'
    elif diagnosis == 'at_risk':
        return 'В зоне риска'
    else:
        return 'Здоров'

def render_template(template: jinja2.Template, parameters: dict, template_files_dir: str):
    return template.render(
        recommendation = parameters['recommendation'],
        user_id = parameters['user_id'],
        issmoke = bool2str(parameters['issmoke']),
        age = parameters['age'],
        duration_audio = parameters['duration_audio'],
        samplerate = parameters['samplerate'],
        commentary = parameters['commentary'],
        diagnosis = diagnosis2str(parameters['diagnosis']),
        episodes = parameters['episodes'],
        intensity = intensity2str(parameters['intensity']),
        productivity = productivity2str(parameters['productivity']),
        sum_probability = round(parameters['sum_probability'], 3),
        main = os.path.join(template_files_dir, 'img/main.jpg'),
        spectrogram = os.path.join(parameters['temp_dir'], 'spectr.png'),
        attention = os.path.join(parameters['temp_dir'], 'attention.png'),
        episodes_spectre = os.path.join(parameters['temp_dir'], 'episodes_spectre.png'),
        # montserrat_100 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-100.svg'),
        # montserrat_200 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-200.svg'),
        # montserrat_300 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-300.svg'),
        # montserrat_400 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-regular.svg'),
        # montserrat_500 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-500.svg'),
        # montserrat_600 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-600.svg'),
        # montserrat_700 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-700.svg'),
        # montserrat_800 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-800.svg'),
        # montserrat_900 = os.path.join(template_files_dir, 'fonts/Montserrat/montserrat-v14-latin-ext_latin_cyrillic-ext_cyrillic-900.svg'),
    )
