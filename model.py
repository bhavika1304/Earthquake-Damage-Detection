import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import math


# ---------------------------
# ECA
# ---------------------------
class ECA(nn.Module):
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        # Adaptive kernel size from channel dimension
        t = int(abs(math.log2(ch) / gamma) + b / gamma)
        k = t if t % 2 else t + 1  # ensure odd

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * self.sigmoid(y).expand_as(x)


# ---------------------------
# CBAM
# ---------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_ = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + max_)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_ = x.max(dim=1, keepdim=True).values
        attn = self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ---------------------------
# Siamese shared ResNet-18 encoder
# ---------------------------
class SiameseEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        res = models.resnet18(weights=weights)
        self.conv1 = nn.Sequential(res.conv1, res.bn1, res.relu)  # /2
        self.maxpool = res.maxpool  # /4
        self.layer1 = res.layer1  # /4  (64)
        self.layer2 = res.layer2  # /8  (128)
        self.layer3 = res.layer3  # /16 (256)
        self.out_channels = [64, 128, 256]

        # ECA@L2 + CBAM@L3
        self.eca2 = ECA(128)
        self.cbam3 = CBAM(256)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.eca2(self.layer2(f1))
        f3 = self.cbam3(self.layer3(f2))  # eca after layer 2 and cbam after layer 3
        return [f1, f2, f3]


# ---------------------------
# Warp helper with cached grid
# ---------------------------
class FeatureWarper(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cached_grid", torch.empty(0))  # initialized lazily

    def forward(self, feat, flow):
        B, C, H, W = feat.shape
        device = feat.device
        # lazily initialize base grid
        if self.cached_grid.numel() == 0 or self.cached_grid.shape[-3:-1] != torch.Size([H, W]):
            yy, xx = torch.meshgrid(
                torch.linspace(0, H - 1, H, device=device),
                torch.linspace(0, W - 1, W, device=device),
                indexing="ij"
            )
            base = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1,2,H,W)
            self.cached_grid = base
        coords = self.cached_grid + flow  # pixel coords
        grid_x = 2.0 * (coords[:, 0, :, :] / (W - 1)) - 1.0
        grid_y = 2.0 * (coords[:, 1, :, :] / (H - 1)) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped


# ---------------------------
# FPFM branch (shared weights)
# ---------------------------
class FPFMBranch(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + x1  # residual
        x3 = self.conv3(x2) + x2  # residual
        return x3


# ---------------------------
# ADFFStage (with FPFM replacing simple subtraction)
# ---------------------------
class ADFFStage(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv_pre = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.conv_post = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.flow_pred = nn.Conv2d(ch * 2, 2, kernel_size=3, padding=1)
        self.warper = FeatureWarper()

        # FPFM: one branch object, called twice with shared weights
        self.branch = FPFMBranch(ch)

        # Final fusion conv
        self.fuse = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch),
                                  nn.ReLU(inplace=True))

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat_pre, feat_post):
        a = self.conv_pre(feat_pre)
        b = self.conv_post(feat_post)
        flow = self.flow_pred(torch.cat([a, b], dim=1))
        a_warp = self.warper(a, flow)
        # Branch I — change magnitude
        branch_I = self.branch(torch.abs(a_warp - b))
        # Branch II — edge consistency (same weights, different input)
        branch_II = self.branch(a_warp + b)

        fc = self.fuse(branch_I + branch_II)
        return fc, flow


# ---------------------------
# Multiscale fusion
# ---------------------------
class MultiscaleFuse(nn.Module):
    def __init__(self, channels_list=[64, 128, 256], out_ch=128):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(ch, out_ch, 1) for ch in channels_list])
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, diffs):
        projected = [p(d) for p, d in zip(self.projs, diffs)]
        target_h, target_w = projected[0].shape[-2:]
        ups = [F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
               for x in projected]
        fused = torch.stack(ups, dim=0).sum(dim=0)
        fused = self.refine(fused)
        return fused


# ---------------------------
# Final Bitemporal model
# ---------------------------
class BitemporalSiamese(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.encoder = SiameseEncoder(pretrained=pretrained, freeze=freeze_encoder)
        chs = self.encoder.out_channels
        self.adffs = nn.ModuleList([ADFFStage(c) for c in chs])
        self.fuse = MultiscaleFuse(chs, out_ch=128)
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
        # init final head
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pre_img, post_img):
        f_pre = self.encoder(pre_img)
        f_post = self.encoder(post_img)
        diffs, flows = [], []
        for fp, fq, ad in zip(f_pre, f_post, self.adffs):
            d, flow = ad(fp, fq)
            diffs.append(d)
            flows.append(flow)
        fused = self.fuse(diffs)
        logits = self.head(fused)
        logits_up = F.interpolate(logits, size=pre_img.shape[-2:], mode='bilinear', align_corners=False)
        return logits_up, {"diffs": diffs, "flows": flows}
