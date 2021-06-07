import torch.nn as nn


class Network(nn.Module):
    def __init__(self, img_ch, net_ch):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(img_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, net_ch, 1, 1, 0)
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(net_ch, net_ch // 2, 1, 1, 0),
            nn.Conv2d(net_ch // 2, img_ch, 1, 1, 0),
            nn.Tanh()
        )
        self.layers = nn.Sequential(
            *[ConvBlock(net_ch, net_ch) for _ in range(6)]
        )

    def forward(self, x):
        x = self.from_rgb(x)

        dense = [x]
        for l in self.layers:
            x = l(x)
            for d in dense:
                x = x + d

        x = self.to_rgb(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, out_c, 3, 1, 0),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.layer(x)
