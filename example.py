import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D


"""
initialize a SwinTransformer3D model
"""
model = SwinTransformer3D(
    pretrained=None,
    pretrained2d=False,
    patch_size=(4, 4, 4),
    # patch_size=(8, 8, 8), # actually increases params by 500k over 4x4x4
    in_chans=12,
    # embed_dim=96,
    embed_dim=96 * 4,
    # depths=[2, 2, 6, 2],
    # num_heads=[3, 6, 12, 24],
    depths=[2, 4, 2],
    num_heads=[3, 6, 12],
    window_size=(2, 7, 7),
    # window_size=(4, 14, 14),
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    norm_layer=nn.LayerNorm,
    patch_norm=False,
    frozen_stages=-1,
    use_checkpoint=False,
)
print(model)
print(sum([param.nelement() for param in model.parameters()]))

# dummy_x = torch.rand(1, 3, 32, 224, 224)
dummy_x = torch.rand(1, 12, 32, 600, 32)
logits = model(dummy_x)
print(logits.shape)

"""
load the pretrained weight

1. git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git
2. move all files into ./Video-Swin-Transformer

"""
# from mmcv import Config, DictAction
# from mmaction.models import build_model
# from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'

# cfg = Config.fromfile(config)
# model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
# load_checkpoint(model, checkpoint, map_location='cpu')

"""
use the pretrained SwinTransformer3D as feature extractor
"""

# [batch_size, channel, temporal_dim, height, width]
dummy_x = torch.rand(1, 3, 32, 224, 224)

# SwinTransformer3D without cls_head
backbone = model.backbone

# [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
feat = backbone(dummy_x)

# alternative way
feat = model.extract_feat(dummy_x)

# mean pooling
feat = feat.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]

# project
batch_size, hidden_dim = feat.shape
feat_dim = 512
proj = nn.Parameter(torch.randn(hidden_dim, feat_dim))

# final output
output = feat @ proj  # [batch_size, feat_dim]
