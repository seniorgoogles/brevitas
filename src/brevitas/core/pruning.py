# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from brevitas.inject.enum import PruningImplType

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

import brevitas
from brevitas import config
from brevitas.core.stats import _ParameterListStats
from brevitas.core.stats import DEFAULT_MOMENTUM
from brevitas.core.stats import SCALAR_SHAPE
from brevitas.function import abs_binary_sign_grad

from .utils import inplace_momentum_update
from .utils import inplace_tensor_add
from .utils import StatelessBuffer

__all__ = [
    'PruningBypass',
    'PruningThreshold',
    'PruningPercentile']


class PruningBypass(brevitas.jit.ScriptModule):

    def __init__(
            self) -> None:
        super(PruningBypass, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        return x
    

class PruningThreshold(brevitas.jit.ScriptModule):

    def __init__(
            self,
            threshold: float) -> None:
        super(PruningThreshold, self).__init__()
        self.threshold = threshold

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        # Set all values below the threshold to zero
        x = torch.where(x.abs() < self.threshold, torch.zeros_like(x), x)
        return x
    
class PruningPercentile(brevitas.jit.ScriptModule):

    def __init__(
            self,
            percentile: float) -> None:
        super(PruningPercentile, self).__init__()
        self.percentile = percentile

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        # Set all values below the threshold to zero
        percentile_value = torch.quantile(abs(x), self.percentile)
        
        x = torch.where(x.abs() <= percentile_value, torch.zeros_like(x), x)
        return x