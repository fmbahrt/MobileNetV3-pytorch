import torch
import torch.nn as nn

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

        # SE
        x = self._squeeze(x)
        x = self._act_depth(x) # activate after SE?

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
    large = [
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
    ]

class MobileNetV3(nn.Module):

    def __init__(self, cfg=None, num_classes=10):
        super(MobileNetV3, self).__init__()

        if cfg is None:
            cfg = Config.large

        # Stem
        self.stem_conv = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(16)
        self.stem_act  = nn.ReLU()

        self.stem = nn.Sequential(
            self.stem_conv,
            self.stem_bn,
            self.stem_act
        )

        # Trunk
        layers = []
        inp = 16
        e_size = 0
        for k_size, e_size, oup, se, swish, stride in cfg:
            block = InvertedResidualBlock(
                inp=inp,
                hiddendim=e_size,
                oup=oup,
                kernel_size=k_size,
                stride=stride,
                se=se,
                swish=swish
            )
            inp = oup
            layers.append(block)
        self.trunk = nn.Sequential(*layers)

        # Classificatio
        c_exp_conv = nn.Conv2d(inp, 960, kernel_size=1, stride=1,
                               padding=0, bias=False)
        c_exp_bn   = nn.BatchNorm2d(960)
        c_exp_act  = nn.Hardswish()

        avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.exp_pooling = nn.Sequential(
            c_exp_conv,
            c_exp_bn,
            c_exp_act,
            avg_pool
        )

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        x = self.exp_pooling(x)
        x = self.classifier(x.view(x.size(0), -1))

        return x

if __name__ == '__main__':
    import time
    x = torch.randn(32, 3, 224, 224)

    model = MobileNetV3()

    start = time.time()
    x = model(x)
    print(x.shape)
    end = time.time() - start

    print(end)

