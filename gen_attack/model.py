import torch.nn as nn


class GivNet(nn.Module):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)

    def __init__(self):
        super(GivNet, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),  # 14 * 14
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),

            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),  # 28 * 28
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),

            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),  # 56 * 56
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),

            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 112 * 112
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 224 * 224
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),

            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 2048, 7, 7)
        out = self.conv(out)
        return out
