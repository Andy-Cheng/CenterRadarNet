import numpy as np
from torch import nn

from ..registry import BACKBONES


@BACKBONES.register_module
# write me a transformer based backbone taking in 3D volumetric data