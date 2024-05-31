# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.sparse import SparseType
from brevitas.inject import ExtendedInjector
from brevitas.inject import this
from brevitas.inject import value
from brevitas.proxy import WeightSparseProxyFromInjector
from brevitas.quant.solver.common import *
from brevitas.sparse.solver.parameter import *

__all__ = [
    'SolveWeightTensorSparseFromEnum',
    'WeightSparseSolver',]
class SolveWeightTensorSparseFromEnum(SolveIntQuantFromEnum):

    @value
    def tensor_sparse(sparse_type):
        if sparse_type == SparseType.NONE:
            return None
        elif sparse_type == SparseType.EPS:
            return None
        elif sparse_type == SparseType.RANDOM:
            return None
        else:
            raise RuntimeError(f'{sparse_type} not recognized.')

class WeightSparseSolver(SolveWeightTensorSparseFromEnum):
    """
    Translate enum and shape directives to weight-specific quantization core modules.
    It should be placed last in the list of classes a quantizer inherits from,
    to make sure overrides are correctly captured.
    """
    proxy_class = WeightSparseProxyFromInjector