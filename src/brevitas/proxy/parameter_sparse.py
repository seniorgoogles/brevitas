# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas import config
from brevitas.function import max_int
from brevitas.quant_tensor import QuantTensor

from .sparse_proxy import SparseProxyFromInjector
from .sparse_proxy import SparseProxyProtocol

__all__ = ['WeightSparseProxyProtocol',
           'ParameterSparseProxyFromInjector']

@runtime_checkable
class WeightSparseProxyProtocol(SparseProxyProtocol, Protocol):

    def forward(self, x: torch.Tensor) -> QuantTensor:
        ...

class ParameterSparseProxyFromInjector(SparseProxyFromInjector):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def tracked_parameter_list(self):
        pass

    def init_tensor_sparse(self, preserve_state_dict=False):
        param_list = self.tracked_parameter_list

        # params might not be there yet, e.g. bias before merging
        if param_list:
            if preserve_state_dict:
                reinit_on_state_dict = config.REINIT_ON_STATE_DICT_LOAD
                ignore_missing_key = config.IGNORE_MISSING_KEYS
                config.REINIT_ON_STATE_DICT_LOAD = False
                config.IGNORE_MISSING_KEYS = True
                state_dict = self.state_dict()
            self.sparse_injector = self.sparse_injector.let(tracked_parameter_list=param_list)
            super(ParameterSparseProxyFromInjector, self).init_tensor_sparse()
            if preserve_state_dict:
                self.load_state_dict(state_dict)
                config.IGNORE_MISSING_KEYS = ignore_missing_key
                config.REINIT_ON_STATE_DICT_LOAD = reinit_on_state_dict

    def max_uint_value(self, bit_width):
        return max_int(False, self.is_narrow_range, bit_width)

class WeightSparseProxyFromInjector(ParameterSparseProxyFromInjector, WeightSparseProxyProtocol):


    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    """
    @property
    def requires_quant_input(self):
        return False

    def scale(self):
        scale = self.__call__(self.tracked_parameter_list[0]).scale
        return scale

    def zero_point(self):
        zero_point = self.__call__(self.tracked_parameter_list[0]).zero_point
        return zero_point

    def bit_width(self):
        bit_width_ = self.__call__(self.tracked_parameter_list[0]).bit_width
        return bit_width_
    """
    def forward(self, x: torch.Tensor) -> QuantTensor:
        if self.is_sparse_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            #impl(x)
            #return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
            return QuantTensor(x, training=self.training)
        else:  # quantization disabled
            return QuantTensor(x, training=self.training)
