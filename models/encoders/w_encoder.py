import math
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class WEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(WEncoder, self).__init__()
        print('Using WEncoder opts.output_size', opts.output_size)
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.output_size, 2))
        self.style_count = 2 * log_size - 2

# WEncoder  forward xx`  torch.Size([1, 3, 256, 256])
# WEncoder forward xx1 result torch.Size([1, 64, 256, 256])
# WEncoder forward xx2 result torch.Size([1, 512, 16, 16])
# WEncoder forward xx3 result torch.Size([1, 512, 1, 1])
# WEncoder forward xx4 result torch.Size([1, 512])
# WEncoder forward xx5 result torch.Size([1, 512])
# WEncoder forward xx6 result torch.Size([1, 18, 512])
    def forward(self, x):
        # [1, 3, 256, 256]
        print("WEncoder  forward xx` ", x.shape)
        x = self.input_layer(x)
        print("WEncoder forward xx1 result", x.shape)
        x = self.body(x)
        print("WEncoder forward xx2 result", x.shape)
        x = self.output_pool(x)
        print("WEncoder forward xx3 result", x.shape)
        x = x.view(-1, 512)
        print("WEncoder forward xx4 result", x.shape)
        x = self.linear(x)
        print("WEncoder forward xx5 result", x.shape)
        x = x.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # [1, 18, 512]
        print("WEncoder forward xx6 result", x.shape)
        return x
