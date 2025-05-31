import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn import UninitializedBuffer
from torch.nn.parameter import UninitializedParameter

# Code from https://github.com/Kaixhin/Rainbow/blob/master/model.py
#           https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
#           https://colab.research.google.com/github/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb#scrollTo=cIkxzBDlfwFz
#           https://arxiv.org/pdf/1706.10295

class NoisyLinear(nn.Module):
    def __init__(self, in_channels, out_channels, std=0.5):
        super(NoisyLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.std = std
        
        #nn.Parameter ensures it says within state_dict as a learnable parameter
        #register_buffer ensures it says within state_dict as a nonlearnable parameter
        self.weight = nn.Parameter(torch.empty(out_channels,in_channels))
        self.weight_sigma = nn.Parameter(torch.empty(out_channels,in_channels))
        self.register_buffer("weight_epsilon",torch.empty(out_channels,in_channels))
        
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.bias_sigma = nn.Parameter(torch.empty(out_channels))
        self.register_buffer("bias_epsilon",torch.empty(out_channels))
        
        self.reset_paramters()
        self.reset_noise()
        
        
    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight + self.weight_sigma * self.weight_epsilon, self.bias + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight, self.bias)
    
    def scale_noise(self, size):
        x = torch.randn(size, device=self.weight.device)
        return x.sign().mul_(x.abs().sqrt_()) #random formula for factorized noise
    
    def reset_noise(self):
        # print('reset')
        epsilon_in = self.scale_noise(self.in_channels).to(self.weight.device)
        epsilon_out = self.scale_noise(self.out_channels).to(self.weight.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out,epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def reset_paramters(self):
        init_range = 1/math.sqrt(self.in_channels)
        self.weight.data.uniform_(-init_range,init_range)
        self.weight_sigma.data.fill_(self.std/math.sqrt(self.in_channels))
        self.bias.data.uniform_(-init_range,init_range)
        self.bias_sigma.data.fill_(self.std/math.sqrt(self.out_channels))
        
        
class LazyNoisyLinear(LazyModuleMixin,nn.Module):
    def __init__(self, out_channels, std=0.5):
        super(LazyNoisyLinear, self).__init__()
        self.in_channels = None
        self.out_channels = out_channels
        self.std = std
        self.init = False
        self.weight = UninitializedParameter()
        self.weight_sigma = UninitializedParameter()
        self.register_buffer("weight_epsilon", UninitializedBuffer())
        self.bias = UninitializedParameter()
        self.bias_sigma = UninitializedParameter()
        self.register_buffer("bias_epsilon", UninitializedBuffer())
        
    def initialize_parameters(self, input):
        self.in_channels = input.size(-1)
        device = input.device
        #nn.Parameter ensures it says within state_dict as a learnable parameter
        #register_buffer ensures it says within state_dict as a nonlearnable parameter
        if isinstance(self.weight, UninitializedParameter):
            self.weight.materialize((self.out_channels, self.in_channels))
            self.weight_sigma.materialize((self.out_channels, self.in_channels))
            self.bias.materialize(self.out_channels)
            self.bias_sigma.materialize(self.out_channels)

        if isinstance(self.weight_epsilon, UninitializedBuffer):
            self.weight_epsilon.materialize((self.out_channels, self.in_channels))
            self.bias_epsilon.materialize(self.out_channels)
        
        self.init = True
        self.reset_paramters()
        self.reset_noise()
        
        
        
    def forward(self, x):
        if not self.init:
            # print(x.size(-1),x.device)
            self.initFunc(x.size(-1),x.device)
        if self.training:
            return F.linear(x, self.weight + self.weight_sigma * self.weight_epsilon, self.bias + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight, self.bias)
    
    def scale_noise(self, size):
        x = torch.randn(size, device=self.weight.device)
        return x.sign().mul_(x.abs().sqrt_()) #random formula for factorized noise
    
    def reset_noise(self):
        if not self.init:
            print('reset_noise called before init')
            return
        # print('lazyreset')
        epsilon_in = self.scale_noise(self.in_channels).to(self.weight.device)
        epsilon_out = self.scale_noise(self.out_channels).to(self.weight.device)
        self.weight_epsilon.copy_(torch.outer(epsilon_out,epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def reset_paramters(self):
        if not self.init:
            print('reset_paramters called before init')
            return
        init_range = 1/math.sqrt(self.in_channels)
        self.weight.data.uniform_(-init_range,init_range)
        self.weight_sigma.data.fill_(self.std/math.sqrt(self.in_channels))
        self.bias.data.uniform_(-init_range,init_range)
        self.bias_sigma.data.fill_(self.std/math.sqrt(self.out_channels))
        