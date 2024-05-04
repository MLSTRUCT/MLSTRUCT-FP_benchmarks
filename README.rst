
======================
MLSTRUCT_FP-benchmarks
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

Benchmarks of MLStructFP dataset.


Description
-----------

This repo contains the segmentation and vectorization models for processing our
`MLSTRUCT-FP dataset <https://github.com/MLSTRUCT/MLSTRUCT-FP>`_. See the following
jupyter notebook files (in order) for more information and a quick start:

- `create_data <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/create_data.ipynb>`_: Creates a dataset, assembles crops, and export data session
- `fp_unet <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/fp_unet.ipynb>`_: Creates U-Net model for wall segmentation
- `vectorization <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/vectorization.ipynb>`_: Vectorizes a model using Egiazarian et al. method


Citing
------

.. code-block:: tex
    
    @article{Pizarro2023,
        title = {Large-scale multi-unit floor plan dataset for architectural plan analysis and recognition},
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

`Pablo Pizarro R. <https://ppizarror.com>`_ | 2023 - 2024
