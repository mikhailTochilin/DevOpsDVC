
from pathlib import Path

from setuptools import setup

NAME = 'enhancer_toch'
DESCRIPTION = (
    'Speech enhancement in the waveform domain.'
    ' Supports offline and streaming evaluation.'
    ' For training, please directly clone the github repository.')

URL = 'https://github.com/mikhailTochilin/DevOpsCI'
EMAIL = 'tochilin.mi@phystech.edu'
AUTHOR = 'Tochilin Mikhail'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.1.0"

HERE = Path(__file__).parent

REQUIRED = [
    'julius',
    'hydra_core<1.0',
    'hydra_colorlog<1.0',
    'numpy>=1.19',
    'pystoi>=0.3.3',
    'six',
    'sounddevice>=0.4',
    'torch>=1.5',
    'torchaudio>=0.5',
]

REQUIRED_LINKS = [
    "git+https://github.com/ludlows/python-pesq#egg=pesq",
]

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=['enhancer_toch'],
    install_requires=REQUIRED,
    dependency_link=REQUIRED_LINKS,
    include_package_data=True,
    license='Public Domain',
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
