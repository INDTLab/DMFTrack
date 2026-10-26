import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.layers.frozen_bn import FrozenBatchNorm2d

class DynamicMF(nn.Module): # 动态加权多尺度特征融合 (多尺度特征融合 + 动态门控)
    """ Multi-scale feature fusion with Dynamic Gating """
    def __init__(self, in_channels=384, scales=[0.8, 1.0, 1.2], reduction=4, freeze_bn=False):
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        self.in_channels = in_channels
        out_channels = in_channels // reduction

        # 多尺度卷积处理
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                FrozenBatchNorm2d(out_channels) if freeze_bn else nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in scales
        ])

        # 动态 gate 网络：从全局 pooled feature → 输出 [B, S] 权重
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, self.num_scales)
        )

        # 融合映射回原通道
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, 1),
            FrozenBatchNorm2d(in_channels) if freeze_bn else nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ x: [B, C, H, W] """
        B, C, H, W = x.shape
        multi_scale_feats = []

        for scale, conv in zip(self.scales, self.convs):
            if scale == 1.0:
                feat = conv(x)
            else:
                scaled_size = (max(8, int(H * scale)), max(8, int(W * scale)))
                scaled_x = F.interpolate(x, size=scaled_size, mode='bilinear', align_corners=False)
                feat = conv(scaled_x)
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            multi_scale_feats.append(feat)

        # [B, S, C', H, W]
        stacked_feats = torch.stack(multi_scale_feats, dim=1)

        # 动态计算 gate 权重
        pooled = F.adaptive_avg_pool2d(x, 1).view(B, C)  # [B, C]
        gate_logits = self.gate_mlp(pooled)              # [B, S]
        gate_weights = F.softmax(gate_logits, dim=1).view(B, self.num_scales, 1, 1, 1)

        # 加权融合
        fused_feat = (stacked_feats * gate_weights).sum(dim=1)  # [B, C', H, W]
        out = self.fusion(fused_feat)

        return out + x  # 残差连接