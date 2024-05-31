# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional

from torch import nn
from torch import tensor
from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from brevitas import config
from brevitas.common import ExportMixin
from brevitas.core.utils import StatelessBuffer
from brevitas.inject import BaseInjector as Injector
from brevitas.utils.quant_utils import float_to_int_impl_to_enum

__all__ = [
    'SparseProxyProtocol',
    'SparseProxyFromInjector',]

"""
def _is_groupwise(sparse_injector):
    if 'group_size' in sparse_injector:
        return True
    else:
        return False


def _is_signed(sparse_injector):
    if 'signed' in sparse_injector:
        return sparse_injector.signed
    return None


def _is_narrow_range(sparse_injector):
    if 'narrow_range' in sparse_injector:
        return sparse_injector.narrow_range
    return None


def _rounding_mode(sparse_injector):
    if 'float_to_int_impl_type' in sparse_injector:
        return str(sparse_injector.float_to_int_impl_type)
    elif 'float_to_int_impl' in sparse_injector:
        try:
            impl_type = float_to_int_impl_to_enum(sparse_injector.float_to_int_impl)
            return str(impl_type).upper()
        except:
            return None
    else:
        return None
"""

def _update_state_dict_impl(quant_injector):
    try:
        impl = quant_injector.update_state_dict_impl
    except:
        impl = None
    return impl


@runtime_checkable
class SparseProxyProtocol(Protocol):
    is_sparse_enabled: bool
    #is_signed: Optional[bool]
    #is_narrow_range: Optional[bool]
    #rounding_mode: Optional[str]
    sparse_first: Optional[bool]

    def add_tracked_module(self, module: nn.Module) -> None:
        ...


class SparseProxyFromInjector(ExportMixin, nn.Module, SparseProxyProtocol):
    __metaclass__ = ABCMeta
    #@todo: Implement the SparseProxyFromInjector class
    def __init__(self, quant_layer: nn.Module, sparse_injector: Injector) -> None:
        ExportMixin.__init__(self)
        nn.Module.__init__(self)
        SparseProxyProtocol.__init__(self)
        self.update_state_dict_impl = _update_state_dict_impl(sparse_injector)
        self.sparse_injector = sparse_injector
        self.sparse_injector = sparse_injector.let(proxy_module=self)
        #self._zero_hw_sentinel = StatelessBuffer(tensor(0.0))
        self.tensor_quant = None
        # Use a normal list and not a ModuleList since this is a pointer to parent modules
        self.tracked_module_list = []
        self.add_tracked_module(quant_layer)
        self.disable_sparse = False

    @property
    def requires_export_handler(self):
        return self.is_sparse_enabled

    def update_tracked_modules(self):
        """Update the modules tracked by the injector with the modules tracked by the proxy"""
        if len(self.tracked_module_list) == 1:
            self.sparse_injector = self.sparse_injector.let(module=self.tracked_module_list[0])
        else:
            # set the list in the injector as a tuple to avoid dealing with inplace modifications
            self.sparse_injector = self.sparse_injector.let(module=tuple(self.tracked_module_list))

    def init_tensor_sparse(self):
        self.tensor_sparse = self.sparse_injector.tensor_sparse

    """
    @property
    def sparse_first(self):
        return self.sparse_first

    @property
    def is_signed(self):
        return _is_signed(self.sparse_injector)

    @property
    def is_groupwise(self):
        return _is_groupwise(self.sparse_injector)

    @property
    def is_narrow_range(self):
        return _is_narrow_range(self.sparse_injector)

    @property
    def rounding_mode(self):
        return _rounding_mode(self.sparse_injector)
    """
    @property
    def is_sparse_enabled(self):
        return not self.disable_sparse and self.tensor_sparse is not None

    def add_tracked_module(self, module: nn.Module) -> None:
        if module is not None:
            self.tracked_module_list.append(module)
            self.update_tracked_modules()
            self.init_tensor_sparse()
        else:
            raise RuntimeError("Trying to add None as a parent module.")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,error_msgs):
        if self.update_state_dict_impl is not None:
            self.update_state_dict_impl(prefix, state_dict)
        super(SparseProxyFromInjector, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # reload tensor_quant on changes of the state_dict
        # this is called after the parent module state_dict is restored (e.g. weights)
        # so init_tensor_quant takes into account new data from the parent module,
        # but before the state_dict of tensor_quant is loaded, so in case e.g. there is a value
        # for the parameter already, it's not overwritten
        if config.REINIT_ON_STATE_DICT_LOAD:
            self.init_tensor_quant()
        # for retrocompatibility with when it wasn't removed
        #zero_hw_sentinel_key = prefix + 'zero_hw_sentinel'
        #if zero_hw_sentinel_key in unexpected_keys:
        #    unexpected_keys.remove(zero_hw_sentinel_key)
