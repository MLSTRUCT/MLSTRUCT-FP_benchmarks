
======================
MLSTRUCT-FP_benchmarks
======================

.. image:: https://img.shields.io/github/actions/workflow/status/MLSTRUCT/MLSTRUCT-FP_benchmarks/ci.yml?branch=master
    :target: https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/actions/workflows/ci.yml
    :alt: Build status

.. image:: https://img.shields.io/github/issues/MLSTRUCT/MLSTRUCT-FP_benchmarks
    :target: https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/issues
    :alt: Open issues

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License MIT

Benchmarks of `MLSTRUCT-FP <https://github.com/MLSTRUCT/MLSTRUCT-FP>`_ dataset.


Description
-----------

This repo contains the segmentation and vectorization models for processing our
`MLSTRUCT-FP dataset <https://github.com/MLSTRUCT/MLSTRUCT-FP>`_. See the following
jupyter notebook files for more information and a quick start (in order):

- `create_data <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/create_data.ipynb>`_: Creates a dataset, assembles crops, and export data session
- `fp_unet <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/fp_unet.ipynb>`_: Creates U-Net model for wall segmentation
- `vectorization <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/vectorization.ipynb>`_: Vectorizes a model using Egiazarian et al. method

The weights for the best model (no_rot_256_50) can be downloaded at
`this link <https://drive.google.com/file/d/15ufkjoWOFyT0Cm-MEc9zQJCDJIooOgh7/view?usp=sharing>`_. For the vectorization model, follow the following links
to download the weights for `model_curves <https://drive.google.com/file/d/18jN37pMvEg9S05sLdAznQC5UZDsLz-za/view?usp=sharing>`_ and
`model_lines <https://drive.google.com/file/d/1Zf085V3783zbrLuTXZxizc7utszI9BZR/view?usp=sharing>`_; check the vectorization
`original repo <https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings>`_ for more details.


First steps
-----------

To use this code, you need a Python 3.8 installation. Then, run the following setup (assuming conda manager):

.. code-block:: bash

    # Clone this repo
    git clone https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks.git
    cd MLSTRUCT-FP_benchmarks

    # Create conda environment & install deps.
    conda create -n mlstructfp python=3.8
    conda activate mlstructfp
    pip install -e .
    conda install jupyter

    # Run the notebook
    jupyter notebook

    # Optional: Install CUDA toolkit if Tensorflow cannot detect GPU
    conda install cudatoolkit=10.1 cudnn=7.6 -c conda-forge

CUDA 10.1 is required to run the trained models.


Citing
------

.. code-block:: tex
    
    @article{Pizarro2023,
      title = {Large-scale multi-unit floor plan dataset for architectural plan analysis and
               recognition},
      journal = {Automation in Construction},
      volume = {156},
      pages = {105132},
      year = {2023},
      issn = {0926-5805},
      doi = {https://doi.org/10.1016/j.autcon.2023.105132},
      url = {https://www.sciencedirect.com/science/article/pii/S0926580523003928},
      author = {Pablo N. Pizarro and Nancy Hitschfeld and Ivan Sipiran}
    }


Author
------

`Pablo Pizarro R. <https://ppizarror.com>`_ | 2023 - 2025
