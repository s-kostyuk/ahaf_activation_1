import torch
from typing import Tuple


class AHAF(torch.nn.Module):
    def __init__(self, *, size: Tuple[int, ...] = (1,), init_as: str = 'ReLU'):
        super(AHAF, self).__init__()

        if init_as == 'ReLU':
            self.gamma = torch.nn.Parameter(torch.ones(*size)*1e9)
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'SiL':
            self.gamma = torch.nn.Parameter(torch.ones(*size))
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'CUSTOM':
            self.gamma = torch.nn.Parameter(torch.ones(*size)*10)
            self.beta = torch.nn.Parameter(torch.ones(*size))
        else:
            raise ValueError("Invalid initialization mode [{}]".format(init_as))

    def forward(self, inputs):
        sig_in = self.gamma * inputs
        sig_out = torch.sigmoid(sig_in)
        amplified = inputs * self.beta
        out = sig_out * amplified
        return out

    # TODO: Verify that the backward pass is implemented by autograd
