
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels) ** 0.5

        self.weight1 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes, modes,
                dtype=torch.cfloat
            )
        )
        self.weight2 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes, modes,
                dtype=torch.cfloat
            )
        )

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch,
            self.weight1.shape[1],
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        m = min(self.modes, height // 2, width // 2 + 1)

        out_ft[:, :, :m, :m] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m, :m],
            self.weight1[:, :, :m, :m]
        )

        out_ft[:, :, -m:, :m] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -m:, :m],
            self.weight2[:, :, :m, :m]
        )

        return torch.fft.irfft2(out_ft, s=(height, width))


class FNO2d(nn.Module):
    def __init__(self, in_channels=5, width=24, modes=8):
        super().__init__()

        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec1 = SpectralConv2d(width, width, modes)
        self.spec2 = SpectralConv2d(width, width, modes)
        self.local1 = nn.Conv2d(width, width, 1)
        self.local2 = nn.Conv2d(width, width, 1)

        self.out = nn.Sequential(
            nn.Conv2d(width, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        x = F.gelu(self.spec1(x) + self.local1(x))
        x = F.gelu(self.spec2(x) + self.local2(x))
        return self.out(x)


class AMFNO2d(nn.Module):
    def __init__(self, in_channels=5, width=24, modes=8):
        super().__init__()

        self.source_weights = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, in_channels, 1)
        )

        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec1 = SpectralConv2d(width, width, modes)
        self.spec2 = SpectralConv2d(width, width, modes)
        self.local1 = nn.Conv2d(width, width, 1)
        self.local2 = nn.Conv2d(width, width, 1)

        self.out = nn.Sequential(
            nn.Conv2d(width, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.source_weights(x), dim=1)
        x = x * weights

        x = self.lift(x)
        x = F.gelu(self.spec1(x) + self.local1(x))
        x = F.gelu(self.spec2(x) + self.local2(x))

        return self.out(x)
