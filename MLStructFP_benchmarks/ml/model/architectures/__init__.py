"""
MLSTRUCTFP BENCHMARKS - ML - MODEL - ARCHITECTURES

Model definition.
"""

from MLStructFP_benchmarks.ml.model.architectures._data_floor_photo_xy import DataFloorPhotoXY, \
    load_floor_photo_data_from_session

from MLStructFP_benchmarks.ml.model.architectures._fp_pix2pix import Pix2PixFloorPhotoModel
from MLStructFP_benchmarks.ml.model.architectures._fp_pix2pix_mod import Pix2PixFloorPhotoModModel
from MLStructFP_benchmarks.ml.model.architectures._fp_pix2pix_patchGAN import Pix2PixPatchGANFloorPhotoModel
from MLStructFP_benchmarks.ml.model.architectures._fp_unet import UNETFloorPhotoModel
