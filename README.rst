
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
jupyter notebook files for more information and quick start:

- `create_data <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/create_data.ipynb>`_: Creates a dataset, assemble crops, and export data session
- `fp_unet <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/fp_unet.ipynb>`_: Creates U-Net model for wall segmentation
- `vectorization <https://github.com/MLSTRUCT/MLSTRUCT-FP_benchmarks/blob/master/vectorization.ipynb>`_: Vectorizes a model using Egiazarian et al. method


Author
------

`Pablo Pizarro R. <https://ppizarror.com>`_ | 2023
