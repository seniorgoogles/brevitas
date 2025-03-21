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

    def __init__(self, 
                 # Todo: Adding ShiftAdd Scheme here like for pruning
                 # int_quant: Module,
                 scaling_impl: Module,
                 #zero_point_impl: Module,
                 bit_width_impl: Module,
                 quant_delay_steps: int = 0,
                 shiftadd_bit_width: int = 10,
                 shiftadd_cost: int = 1,
                 ):
        super(ShiftAddQuant, self).__init__()
        self.scaling_impl = scaling_impl
        self.zero_point = StatelessBuffer(torch.tensor(0.0))
        self.bit_width_impl = bit_width_impl
        self.delay_wrapper = DelayWrapper(quant_delay_steps)
        self.shiftadd_bit_width = shiftadd_bit_width
        self.shiftadd_cost = shiftadd_cost

        def generate_shiftadd_values(shiftadd_bit_width: int, shiftadd_cost: int):
            
            cost_list = []
            
            if shiftadd_cost == 0:
                cost_list=[1,2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                # Add negatives 
                cost_list = cost_list + [-i for i in cost_list]
                # Add 0
                cost_list.append(0)
                cost_list.sort()
            elif shiftadd_cost == 1:
                cost_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 24, 28, 30, 31, 32, 33, 34, 36, 40, 48, 56, 60, 62, 63, 64, 65, 66, 68, 72, 80, 
                           96, 112, 120, 124, 126, 127, 128, 129, 130, 132, 136, 144, 160, 192, 224, 240, 248, 252, 254, 255, 256, 257, 258, 260, 264, 272, 288, 320, 
                           384, 448, 480, 496, 504, 508, 510, 511, 512, 513, 514, 516, 520, 528, 544, 576, 640, 768, 896, 960, 992, 1008, 1016, 1020, 1022, 1023, 1024]
                # Add negatives 
                cost_list = cost_list + [-i for i in cost_list]
                # Add 0
                cost_list.append(0)
                cost_list.sort()
                
            else:
                raise Exception("ShiftAdd cost must be 0 or 1")
            
            return cost_list
        # Generate allowed values
        allowed_values = generate_shiftadd_values(self.shiftadd_bit_width, self.shiftadd_cost)
        
        # Register allowed_values as a buffer so that it moves with the module.
        allowed_tensor = torch.tensor(allowed_values)
        self.register_buffer("allowed_values", allowed_tensor)

        if self.allowed_values.numel() == 0:
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
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        scale = self.scaling_impl(x)
        zero_point = self.zero_point()
        bit_width = self.bit_width_impl()
        
        #print(f"Scale: {scale}")
        #print(f"Zero Point: {zero_point}")
        #print(f"Bit Width: {bit_width}")
        #print(f"Allowed Values: {self.allowed_values}")
        
        min_input = torch.min(x)
        max_input = torch.max(x)
    
        scale = (max_input - min_input) / ((2 ** bit_width) - 1)
        #print(f"Scale: {scale}")
                
        y = self.quantize_to_array(x, self.allowed_values * scale) # todo add scaling with: self.allowed_values * scale
        
        return y, scale, zero_point, bit_width


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
