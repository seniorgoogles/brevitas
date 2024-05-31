from brevitas.core.bit_width import *
from brevitas.core.function_wrapper import *
from brevitas.core.function_wrapper.learned_round import LearnedRoundHardSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSigmoid
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.core.function_wrapper.stochastic_round import StochasticRoundSte
from brevitas.core.sparse import *
from brevitas.core.sparse import SparseType
from brevitas.core.restrict_val import *
from brevitas.core.scaling import *
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import *
from brevitas.inject import ExtendedInjector
from brevitas.inject import value
from brevitas.inject.enum import LearnedRoundImplType
class SolveSparseFromEnum(ExtendedInjector):

    @value
    def sparse(sparse_type):
        if sparse_type == SparseType.NONE:
            return None
        elif sparse_type == SparseType.EPS:
            return None
        elif sparse_type == SparseType.RANDOM:
            return None
        else:
            return None