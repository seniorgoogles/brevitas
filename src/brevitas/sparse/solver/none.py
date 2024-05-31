from brevitas.inject.enum import SparseType
from brevitas.sparse.solver import WeightSparseSolver

__all__ = ['NoneWeightSparse']

class NoneWeightSparse(WeightSparseSolver):
    """
    Base quantizer used when weight_sparse_quant=None.
    """
    sparse_type = SparseType.NONE