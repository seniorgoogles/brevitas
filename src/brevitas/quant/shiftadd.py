from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant.delay import DelayWrapper
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops_ste import binary_sign_ste

class ShiftAddQuant(brevitas.jit.ScriptModule):

    def __init__(self, scaling_impl: Module, quant_delay_steps: int = 0, allowed_values=[]):
        super(ShiftAddQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)
        self.allowed_values = torch.asarray(allowed_values) # todo add as parameter
        if self.allowed_values.size()[0] > 0:
            raise Exception("There are no allowed values to quantize to.")

    def quantize_to_array(self, inputs: Tensor, array: Tensor):
        q_reshaped = torch.reshape(array, [array.shape[0], 1])
        # reshape x to flat tensor
        x_tmp = torch.reshape(inputs, [-1])
        # build a grid of abs differences between x and q
        # find the index of the minimal distance
        abs_diff = torch.abs(x_tmp - q_reshaped)
        min_index = torch.argmin(abs_diff, dim=0)
        # get the corresponding quantisation value
        y_tmp = torch.gather(array, dim=0, index=min_index)
        # reshape the result back to its oroginal form
        y = torch.reshape(y_tmp, inputs.shape)
        return y

    # Forward path quantizer
    @brevitas.jit.script_method
    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(inputs)
        print(scale)
        # y = x * 0.0001

        y = self.quantize_to_array(inputs, self.allowed_values * scale) # todo add scaling with: self.allowed_values * scale

        return y, scale, self.zero_point(), self.bit_width()


class ClampedShiftAddQuant(brevitas.jit.ScriptModule):

    def __init__(
            self,
            scaling_impl: Module,
            tensor_clamp_impl: Module = TensorClamp(),
            quant_delay_steps: int = 0):
        super(ClampedShiftAddQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.bit_width = BitWidthConst(1)
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.delay_wrapper = DelayWrapper(quant_delay_steps)
        self.tensor_clamp_impl = tensor_clamp_impl

    # Backward path quantizer?
    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        y = self.tensor_clamp_impl(x, -scale, scale)
        y = binary_sign_ste(y) * scale
        y = self.delay_wrapper(x, y)
        return y, scale, self.zero_point(), self.bit_width()
