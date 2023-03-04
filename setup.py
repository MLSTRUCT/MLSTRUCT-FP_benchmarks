"""
MLSTRUCTFP-benchmarks - SETUP

Setup distribution.
"""

# Library imports
import os
import MLStructFP_benchmarks
from setuptools import setup, find_packages

# Load readme
with open('README.rst') as f:
    long_description = f.read()

# Load requirements
requirements = [
    'Keras == 2.3.1',
    'keras_tqdm == 2.0.1',
    'matplotlib == 3.5.3',
    'MLStructFP == 0.3.1',
    'numpy == 1.18.5'
    'scikit-learn == 1.2.0'
]

if os.environ.get('GITHUB') != 'true':
    for r in [
        'tensorboard == 2.2.2',
        'tensorflow-gpu == 2.2.2'  # Check https://www.tensorflow.org/install/gpu. Needs CUDA 10.1 + cuDNN 7.6.5
    ]:
        requirements.append(r)

# Setup library
setup(
    author=MLStructFP_benchmarks.__author__,
    author_email=MLStructFP_benchmarks.__email__,
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    description=MLStructFP_benchmarks.__description__,
    long_description=long_description,
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        'test': ['codecov', 'nose2']
    },
    keywords=MLStructFP_benchmarks.__keywords__,
    name='MLStructFP-benchmarks',
    packages=find_packages(exclude=[
        '.idea',
        '.ipynb_checkpoints',
        'test'
    ]),
    platforms=['any'],
    project_urls={
        'Bug Tracker': MLStructFP_benchmarks.__url_bug_tracker__,
        'Documentation': MLStructFP_benchmarks.__url_documentation__,
        'Source Code': MLStructFP_benchmarks.__url_source_code__
    },
    python_requires='>=3.8',
    url=MLStructFP_benchmarks.__url__,
    version=MLStructFP_benchmarks.__version__
)
