# from copy import deepcopy
import numpy as np

from mrinversion.kernel.lineshape import DAS
from mrinversion.kernel.lineshape import MAF
from mrinversion.kernel.lineshape import NuclearShieldingTensor
from mrinversion.kernel.lineshape import SpinningSidebands
from mrinversion.kernel.lineshape import x_y_to_zeta_eta
from mrinversion.kernel.relaxation import T1
from mrinversion.kernel.relaxation import T2
