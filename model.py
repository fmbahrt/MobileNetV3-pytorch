import torch
import torch.nn as nn

# Utility function for width multiplication
def _make_divisible(v, divisor, min_value=None):
    """
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SEBlock(nn.Module):

    """Sweeet squeeze and excitation module!"""

    def __init__(self, in_channel, ratio=4):
        super(SEBlock, self).__init__()

        self.hid_dim = in_channel // ratio

        self._pool = nn.AdaptiveAvgPool2d(1)
        self._ln1  = nn.Linear(in_channel, self.hid_dim)
        self._ln2  = nn.Linear(self.hid_dim, in_channel)

        self._act1 = nn.ReLU()
        self._act2 = nn.Hardsigmoid() # we dont not like sigmoid hihi :^)

    def forward(self, x):
        b, c, _, _ = x.size() # Squeeze doesnt work if batch dim = 1 :---(

        # FC part
        z = self._pool(x).view(b, c)
        z = self._ln1(z)
        z = self._act1(z)
        z = self._ln2(z)
        z = self._act2(z).view(b, c, 1, 1)

        # Now weigh each channel
        return x * z

class InvertedResidualBlock(nn.Module):

    def __init__(self,
                 inp,
                 hiddendim,
                 oup,
                 kernel_size,
                 stride,
                 se=False,
                 swish=False):
        super(InvertedResidualBlock, self).__init__()

        self.inp = inp
        self.oup = oup
        self.hip = hiddendim
        self.stride = stride

        # Channel Expansion
        self._expand = nn.Conv2d(inp, hiddendim, kernel_size=1, stride=1, bias=False)
        self._bn_exp = nn.BatchNorm2d(hiddendim)
        self._act_exp = nn.Hardswish() if swish else nn.ReLU()

        # Depthwise Convolution
        self._depth = nn.Conv2d(in_channels=hiddendim, out_channels=hiddendim,
                                groups=hiddendim, kernel_size=kernel_size,
                                stride=stride, padding=(kernel_size-1)//2, bias=False)
        self._bn_depth = nn.BatchNorm2d(hiddendim)
        self._act_depth = nn.Hardswish() if swish else nn.ReLU()

        # Squeeze and excitation
        self._squeeze = SEBlock(hiddendim) if se else nn.Identity()

        # Projection
        self._proj = nn.Conv2d(in_channels=hiddendim, out_channels=oup,
                               kernel_size=1, stride=1, bias=False)
        self._bn_proj = nn.BatchNorm2d(oup)

    def forward(self, x):
        ins = x.clone()

        # No reason to expand if #in channels is equal #hidden channels
        if self.inp != self.hip:
            # Expand
            x = self._expand(x)
            x = self._bn_exp(x)
            x = self._act_exp(x)

        # Depthwise Conv
        x = self._depth(x)
        x = self._bn_depth(x)
        x = self._act_depth(x)

        # SE
        x = self._squeeze(x)

        # Project
        x = self._proj(x)
        x = self._bn_proj(x)

        # If input dim equal output dim then we use a residual connection
        #  as stated in mobilenetv2 paper
        if self.inp == self.oup and self.stride == 1:
            x = ins + x

        return x

class Config():

    # kernel_size, exp_size, out_channels, se, swish, stride

    # Ref: table 1 MobileNetV3 paper
    large = {
        'blocks': [
            [3, 16, 16, False, False, 1],
            [3, 64, 24, False, False, 2],
            [3, 72, 24, False, False, 1],
            [5, 72, 40, True, False, 2],
            [5, 120, 40, True, False, 1],
            [5, 120, 40, True, False, 1],
            [3, 240, 80, False, True, 2],
            [3, 200, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 184, 80, False, True, 1],
            [3, 480, 112, True, True, 1],
            [3, 672, 112, True, True, 1],
            [5, 672, 160, True, True, 2],
            [5, 960, 160, True, True, 1],
            [5, 960, 160, True, True, 1],
        ],
        'out_width': 960,
        'classification_width': 1280
    }

    small = {
        'blocks': [
            [3, 16, 16, True, False, 2],
            [3, 72, 24, False, False, 2],
            [3, 88, 24, False, False, 1],
            [5, 96, 40, True, True, 2],
            [5, 240, 40, True, True, 1],
            [5, 240, 40, True, True, 1],
            [5, 120, 48, True, True, 1],
            [5, 144, 48, True, True, 1],
            [5, 288, 96, True, True, 2],
            [5, 576, 96, True, True, 1],
            [5, 576, 96, True, True, 1],
        ],
        'out_width': 576,
        'classification_width': 1024
    }

class MobileNetV3(nn.Module):

    def __init__(self, cfg=None, num_classes=10, alpha=1.):
        super(MobileNetV3, self).__init__()

        if cfg is None:
            cfg = Config.large

        # Stem
        first_out_dim = _make_divisible(16 * alpha, 8)
        self.stem_conv = nn.Conv2d(3, first_out_dim,
                                   kernel_size=1,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.stem_bn   = nn.BatchNorm2d(first_out_dim)
        self.stem_act  = nn.ReLU()

        self.stem = nn.Sequential(
            self.stem_conv,
            self.stem_bn,
            self.stem_act
        )

        # Trunk
        layers = []
        inp = first_out_dim
        e_size = 0
        for k_size, e_size, oup, se, swish, stride in cfg['blocks']:

            # Account for width multiplier
            oup = _make_divisible(oup * alpha, 8)
            exp_size = _make_divisible(e_size * alpha, 8)

            block = InvertedResidualBlock(
                inp=inp,
                hiddendim=exp_size,
                oup=oup,
                kernel_size=k_size,
                stride=stride,
                se=se,
                swish=swish
            )
            inp = oup
            layers.append(block)
        self.trunk = nn.Sequential(*layers)

        # Classification
        c_exp_conv = nn.Conv2d(inp, cfg['out_width'], kernel_size=1, stride=1,
                               padding=0, bias=False)
        c_exp_bn   = nn.BatchNorm2d(cfg['out_width'])
        c_exp_act  = nn.Hardswish()

        avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.exp_pooling = nn.Sequential(
            c_exp_conv,
            c_exp_bn,
            c_exp_act,
            avg_pool
        )

        c_conv1 = nn.Conv2d(cfg['out_width'], cfg['classification_width'], kernel_size=1, stride=1,
                            padding=0, bias=False)
        c_conv2 = nn.Conv2d(cfg['classification_width'], num_classes, kernel_size=1, stride=1,
                            padding=0, bias=False)

        self.classifier = nn.Sequential(
            c_conv1,
            nn.Hardswish(),
            c_conv2
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.exp_pooling(x)
        x = self.classifier(x)

        return torch.squeeze(x)

if __name__ == '__main__':
    import time
    inz = torch.randn(32, 3, 224, 224)

    model = MobileNetV3(cfg=Config.small, alpha=0.5)

    start = time.time()
    for i in range(10):
        x = model(inz)
        print(x.shape)
    end = time.time() - start

    print(end)

