import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model
from models.functions import load_scene_model


@Model.register()
class CDMVGGT(nn.Module):
    """3D affordance mask refinement conditioned on lifted VGGT features.

    Same structure as the active CDM (PointTransformerSeg over xyz + initial
    mask), but the per-point input features are the initial mask channel
    concatenated with a learned low-dim projection of the per-point VGGT
    features (``c_vggt_feat``, produced by step7b).
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.contact_dim = cfg.input_feats
        self.vggt_feat_dim = cfg.vggt_feat_dim
        self.vggt_proj_dim = cfg.vggt_proj_dim

        # VGGT tokens come at large scale/width; normalize and project down so
        # they don't drown out the single mask channel
        self.vggt_proj = nn.Sequential(
            nn.LayerNorm(self.vggt_feat_dim),
            nn.Linear(self.vggt_feat_dim, self.vggt_proj_dim, bias=True),
        )

        self.scene_model = load_scene_model(
            cfg.scene_model.name,
            self.contact_dim + self.vggt_proj_dim,
            cfg.scene_model.num_points,
            cfg.scene_model.pretrained_weight,
            freeze=cfg.scene_model.freeze,
        )

        self.contact_layer = nn.Linear(32, 1, bias=True)

    def forward(self, x, **kwargs):
        """ Forward pass of the model.

        Args:
            x: input affordance mask, [bs, num_points, 1]
            kwargs: conditions; requires c_pc_xyz [bs, num_points, 3] and
                c_vggt_feat [bs, num_points, vggt_feat_dim]

        Returns:
            Refined mask logits, [bs, num_points, 1]
        """
        vggt_feat = self.vggt_proj(kwargs['c_vggt_feat'].float())
        x = torch.cat([x, vggt_feat], dim=-1)
        x = self.scene_model((kwargs['c_pc_xyz'], x))
        x = self.contact_layer(x)
        return x
