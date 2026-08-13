import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model
from models.functions import load_scene_model


class SplitInputLinear(nn.Module):
    """Base-preserving input stem with separately trainable VGGT columns.

    This is algebraically equivalent to one bias-free Linear over concatenated
    mask/VGGT channels, but keeping the two weight blocks separate lets a
    mask-only checkpoint initialize the base block exactly while the VGGT block
    starts at zero and can use its own learning rate.
    """

    def __init__(self, base_dim, vggt_dim, out_dim):
        super().__init__()
        self.base_dim = int(base_dim)
        self.vggt_dim = int(vggt_dim)
        self.base = nn.Linear(self.base_dim, out_dim, bias=False)
        self.vggt = nn.Linear(self.vggt_dim, out_dim, bias=False)
        nn.init.zeros_(self.vggt.weight)

    def forward(self, features):
        expected = self.base_dim + self.vggt_dim
        if features.shape[-1] != expected:
            raise ValueError(
                f'expected {expected} input channels, got {features.shape[-1]}'
            )
        return (
            self.base(features[..., :self.base_dim])
            + self.vggt(features[..., self.base_dim:])
        )


class PostStemVGGTTransition(nn.Module):
    """Fuse VGGT with the completed mask stem while preserving its identity.

    The wrapped stride-one transition keeps the mask-only stem parameter names
    unchanged for strict checkpoint loading.  Its new fusion layer sees the
    concatenated 32-D mask embedding and VGGT projection, and starts at zero so
    the complete module initially returns the unmodified mask embedding.
    """

    def __init__(self, base_stem, base_dim, vggt_dim):
        super().__init__()
        if base_stem.stride != 1:
            raise ValueError('post-stem VGGT fusion requires a stride-one stem')
        if base_stem.linear.in_features != base_dim:
            raise ValueError(
                'mask stem input width does not match base_dim: '
                f'{base_stem.linear.in_features} != {base_dim}'
            )

        self.stride = base_stem.stride
        self.nsample = base_stem.nsample
        self.base_dim = int(base_dim)
        self.vggt_dim = int(vggt_dim)
        self.linear = base_stem.linear
        self.bn = base_stem.bn
        self.relu = base_stem.relu

        stem_dim = self.linear.out_features
        self.vggt_fusion = nn.Linear(
            stem_dim + self.vggt_dim, stem_dim, bias=True
        )
        nn.init.zeros_(self.vggt_fusion.weight)
        nn.init.zeros_(self.vggt_fusion.bias)

    def forward(self, pxo):
        p, features, o = pxo
        expected = self.base_dim + self.vggt_dim
        if features.shape[-1] != expected:
            raise ValueError(
                f'expected {expected} post-stem input channels, '
                f'got {features.shape[-1]}'
            )

        mask_feature = features[..., :self.base_dim]
        vggt_feature = features[..., self.base_dim:]
        mask_stem = self.relu(self.bn(self.linear(mask_feature)))
        residual = self.vggt_fusion(
            torch.cat([mask_stem, vggt_feature], dim=-1)
        )
        return [p, mask_stem + residual, o]


@Model.register()
class CDMVGGT(nn.Module):
    """3D affordance mask refinement conditioned on lifted VGGT features.

    Same structure as the active CDM (PointTransformerSeg over xyz + initial
    mask), but the per-point input features are the initial mask channel
    concatenated with a gated, learned low-dim projection of the per-point
    VGGT features. The gate is conditioned on the projected feature, normalized
    VGGT confidence, view count, and whether the point was observed.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.contact_dim = cfg.input_feats
        self.vggt_feat_dim = cfg.vggt_feat_dim
        self.vggt_proj_dim = cfg.vggt_proj_dim
        self.split_input_stem = bool(cfg.get('split_input_stem', False))
        self.post_stem_fusion = bool(cfg.get('post_stem_fusion', False))
        self.freeze_base = bool(cfg.get('freeze_base', False))
        self.freeze_batchnorm = bool(cfg.get('freeze_batchnorm', False))
        if self.split_input_stem and self.post_stem_fusion:
            raise ValueError(
                'split_input_stem and post_stem_fusion are mutually exclusive'
            )

        # VGGT tokens come at large scale/width; normalize and project down so
        # they don't drown out the single mask channel
        self.vggt_proj = nn.Sequential(
            nn.LayerNorm(self.vggt_feat_dim),
            nn.Linear(self.vggt_feat_dim, self.vggt_proj_dim, bias=True),
        )
        self.vggt_gate = nn.Linear(
            self.vggt_proj_dim, self.vggt_proj_dim, bias=True
        )
        self.vggt_reliability_gate = nn.Sequential(
            nn.Linear(3, self.vggt_proj_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.vggt_proj_dim, self.vggt_proj_dim, bias=True),
        )

        self.scene_model = load_scene_model(
            cfg.scene_model.name,
            (
                self.contact_dim
                if self.post_stem_fusion
                else self.contact_dim + self.vggt_proj_dim
            ),
            cfg.scene_model.num_points,
            cfg.scene_model.pretrained_weight,
            freeze=cfg.scene_model.freeze,
        )

        if self.post_stem_fusion:
            self.scene_model.enc1[0] = PostStemVGGTTransition(
                self.scene_model.enc1[0],
                base_dim=self.contact_dim,
                vggt_dim=self.vggt_proj_dim,
            )
        elif self.split_input_stem:
            original_stem = self.scene_model.enc1[0].linear
            if original_stem.bias is not None:
                raise ValueError('controlled early fusion requires a bias-free input stem')
            self.scene_model.enc1[0].linear = SplitInputLinear(
                base_dim=self.contact_dim,
                vggt_dim=self.vggt_proj_dim,
                out_dim=original_stem.out_features,
            )

        self.contact_layer = nn.Linear(32, 1, bias=True)

        if self.freeze_base:
            if not (self.split_input_stem or self.post_stem_fusion):
                raise ValueError(
                    'freeze_base=True requires split_input_stem=True or '
                    'post_stem_fusion=True'
                )
            self.scene_model.requires_grad_(False)
            self.contact_layer.requires_grad_(False)
            if self.post_stem_fusion:
                self.scene_model.enc1[0].vggt_fusion.requires_grad_(True)
            else:
                # The new VGGT columns remain trainable while the copied mask
                # column and the rest of PointTransformer stay fixed.
                self.scene_model.enc1[0].linear.vggt.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.scene_model.eval()
            self.contact_layer.eval()
            if self.post_stem_fusion:
                self.scene_model.enc1[0].vggt_fusion.train(mode)
            else:
                self.scene_model.enc1[0].linear.vggt.train(mode)
        elif self.freeze_batchnorm:
            # Fine-tune weights without allowing small per-GPU batches to
            # overwrite the base checkpoint's running statistics.
            for module in self.scene_model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x, **kwargs):
        """ Forward pass of the model.

        Args:
            x: input affordance mask, [bs, num_points, 1]
            kwargs: conditions; requires c_pc_xyz [bs, num_points, 3],
                c_vggt_feat [bs, num_points, vggt_feat_dim], c_vggt_conf
                [bs, num_points], and c_vggt_view_count [bs, num_points]

        Returns:
            Refined mask logits, [bs, num_points, 1]
        """
        raw_vggt_feat = torch.nan_to_num(
            kwargs['c_vggt_feat'].float(),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        vggt_feat = self.vggt_proj(raw_vggt_feat)
        confidence = kwargs['c_vggt_conf'].float()
        view_count = kwargs['c_vggt_view_count'].float()
        if confidence.ndim == vggt_feat.ndim and confidence.shape[-1] == 1:
            confidence = confidence.squeeze(-1)
        if view_count.ndim == vggt_feat.ndim and view_count.shape[-1] == 1:
            view_count = view_count.squeeze(-1)
        expected_shape = vggt_feat.shape[:-1]
        if confidence.shape != expected_shape or view_count.shape != expected_shape:
            raise ValueError(
                'VGGT confidence and view count must match the point dimensions: '
                f'expected {expected_shape}, got {confidence.shape} and '
                f'{view_count.shape}'
            )

        confidence = torch.nan_to_num(
            confidence, nan=0.0, posinf=1e6, neginf=0.0
        ).clamp_min(0.0)
        view_count = torch.nan_to_num(
            view_count, nan=0.0, posinf=1e6, neginf=0.0
        ).clamp_min(0.0)
        seen = (view_count > 0).to(vggt_feat.dtype)

        # Confidence is positive and unbounded, so use log confidence and
        # standardize it over observed points in each sample. Unseen points are
        # kept at zero and are identified explicitly by the `seen` channel.
        log_conf = torch.log(confidence.clamp_min(1.0))
        seen_count = seen.sum(dim=1, keepdim=True).clamp_min(1.0)
        conf_mean = (log_conf * seen).sum(dim=1, keepdim=True) / seen_count
        conf_var = (
            ((log_conf - conf_mean).square() * seen).sum(dim=1, keepdim=True)
            / seen_count
        )
        normalized_log_conf = (
            (log_conf - conf_mean) / torch.sqrt(conf_var + 1e-6)
        ) * seen
        reliability = torch.stack(
            [normalized_log_conf, torch.log1p(view_count), seen], dim=-1
        )

        gate = torch.sigmoid(
            self.vggt_gate(vggt_feat) + self.vggt_reliability_gate(reliability)
        )
        vggt_feat = gate * vggt_feat
        x = torch.cat([x, vggt_feat], dim=-1)
        x = self.scene_model((kwargs['c_pc_xyz'], x))
        x = self.contact_layer(x)
        return x


class VGGTResidualAdapter(nn.Module):
    """Map lifted VGGT point features to a zero-initialized residual."""

    def __init__(self, feat_dim, hidden_dim=128, out_dim=32,
                 use_reliability=False, reliability_mode='relative',
                 absolute_confidence_log1p_mean=0.0,
                 absolute_confidence_log1p_std=1.0,
                 view_count_log1p_mean=0.0,
                 view_count_log1p_std=1.0):
        super().__init__()
        self.use_reliability = use_reliability
        self.reliability_mode = str(reliability_mode)
        if self.reliability_mode not in ('relative', 'enriched'):
            raise ValueError(
                "reliability_mode must be 'relative' or 'enriched', got "
                f'{self.reliability_mode!r}'
            )
        self.reliability_dim = 4 if self.reliability_mode == 'enriched' else 1
        if self.reliability_mode == 'enriched':
            stats = {
                'absolute_confidence_log1p_mean': absolute_confidence_log1p_mean,
                'absolute_confidence_log1p_std': absolute_confidence_log1p_std,
                'view_count_log1p_mean': view_count_log1p_mean,
                'view_count_log1p_std': view_count_log1p_std,
            }
            for name, value in stats.items():
                value = float(value)
                if not torch.isfinite(torch.tensor(value)):
                    raise ValueError(f'{name} must be finite, got {value}')
                if name.endswith('_std') and value <= 0:
                    raise ValueError(f'{name} must be positive, got {value}')
                self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim, bias=True),
            nn.SiLU(),
        )
        # Keep this projection bias-free so zero reliability inputs remain a
        # strict no-op. The legacy relative-confidence adapter has one input;
        # the enriched variant has four reliability inputs.
        self.confidence_proj = nn.Linear(
            self.reliability_dim, hidden_dim, bias=False
        )
        self.output = nn.Linear(hidden_dim, out_dim, bias=True)

        # The complete model initially behaves exactly like the mask-only
        # baseline. Gradients first train this output layer, then propagate into
        # the rest of the adapter as its weights become non-zero.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _point_scalar(value, point_shape, name):
        value = value.float()
        if value.ndim == len(point_shape) + 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != point_shape:
            raise ValueError(
                f'{name} must have shape {point_shape}, got {value.shape}'
            )
        return torch.nan_to_num(
            value, nan=0.0, posinf=1e6, neginf=0.0
        ).clamp_min(0.0)

    def _reliability(self, feature, confidence, view_count=None):
        point_shape = feature.shape[:-1]
        confidence = self._point_scalar(
            confidence, point_shape, 'c_vggt_conf'
        )
        # The legacy ablations define visibility from the lifted feature so
        # their only difference remains the confidence input. The enriched
        # mode additionally verifies that at least one view contributed.
        seen = (feature.abs().sum(dim=-1) > 0).to(feature.dtype)

        if self.reliability_mode == 'enriched':
            if view_count is None:
                raise ValueError(
                    'c_vggt_view_count is required for enriched reliability inputs'
                )
            view_count = self._point_scalar(
                view_count, point_shape, 'c_vggt_view_count'
            )
            # Require both a lifted feature and at least one contributing view.
            seen = seen * (view_count > 0).to(feature.dtype)

        log_conf = torch.log(confidence.clamp_min(1.0))
        seen_count = seen.sum(dim=1, keepdim=True).clamp_min(1.0)
        conf_mean = (log_conf * seen).sum(dim=1, keepdim=True) / seen_count
        conf_var = (
            ((log_conf - conf_mean).square() * seen).sum(dim=1, keepdim=True)
            / seen_count
        )
        normalized_log_conf = (
            (log_conf - conf_mean) / torch.sqrt(conf_var + 1e-6)
        ) * seen
        if self.reliability_mode == 'relative':
            reliability = normalized_log_conf.unsqueeze(-1)
            return reliability, seen

        absolute_log_conf = (
            (torch.log1p(confidence) - self.absolute_confidence_log1p_mean)
            / self.absolute_confidence_log1p_std
        ) * seen
        normalized_log_view_count = (
            (torch.log1p(view_count) - self.view_count_log1p_mean)
            / self.view_count_log1p_std
        ) * seen
        reliability = torch.stack((
            normalized_log_conf,
            absolute_log_conf,
            normalized_log_view_count,
            seen,
        ), dim=-1)
        return reliability, seen

    def forward(self, feature, confidence=None, view_count=None):
        feature = torch.nan_to_num(
            feature.float(), nan=0.0, posinf=1e6, neginf=-1e6
        )
        hidden = self.feature_proj(feature)

        if self.use_reliability:
            if confidence is None:
                raise ValueError(
                    'confidence is required when use_reliability=True'
                )
            reliability, seen = self._reliability(
                feature, confidence, view_count
            )
        else:
            # Zero feature vectors are the extractor's representation for an
            # unobserved point. Keep their adapter residual exactly zero.
            seen = (feature.abs().sum(dim=-1) > 0).to(feature.dtype)
            reliability = feature.new_zeros(
                (*feature.shape[:-1], self.reliability_dim)
            )

        hidden = hidden + self.confidence_proj(reliability)

        return self.output(hidden) * seen.unsqueeze(-1)


class VGGTAdditiveStemAdapter(nn.Module):
    """Project aligned VGGT descriptors into the PointTransformer stem.

    The DPT descriptor is already a nonlinear dense representation, so fusion
    uses only LayerNorm and one bias-free projection. A scalar confidence gate
    controls the complete descriptor instead of independently gating channels.
    The projection starts at zero, making the containing model exactly equal to
    its mask-only checkpoint at initialization while still receiving a direct
    gradient on the projection weights.
    """

    def __init__(self, feat_dim, stem_dim):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.stem_dim = int(stem_dim)
        self.norm = nn.LayerNorm(self.feat_dim)
        self.projection = nn.Linear(
            self.feat_dim, self.stem_dim, bias=False
        )
        nn.init.zeros_(self.projection.weight)

        # Start with a monotonic confidence gate: below-average observations
        # are suppressed and above-average observations are retained. Both
        # scalars remain learnable.
        self.confidence_scale = nn.Parameter(torch.ones(()))
        self.confidence_bias = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _point_scalar(value, point_shape, name):
        value = value.float()
        if value.ndim == len(point_shape) + 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.shape != point_shape:
            raise ValueError(
                f'{name} must have shape {point_shape}, got {value.shape}'
            )
        return torch.nan_to_num(
            value, nan=0.0, posinf=1e6, neginf=0.0
        ).clamp_min(0.0)

    def confidence_gate(self, feature, confidence, view_count):
        point_shape = feature.shape[:-1]
        confidence = self._point_scalar(
            confidence, point_shape, 'c_vggt_conf'
        )
        view_count = self._point_scalar(
            view_count, point_shape, 'c_vggt_view_count'
        )
        seen = (view_count > 0).to(feature.dtype)

        log_confidence = torch.log(confidence.clamp_min(1.0))
        seen_count = seen.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (log_confidence * seen).sum(dim=1, keepdim=True) / seen_count
        variance = (
            ((log_confidence - mean).square() * seen).sum(dim=1, keepdim=True)
            / seen_count
        )
        normalized = (
            (log_confidence - mean) / torch.sqrt(variance + 1e-6)
        ) * seen
        gate = torch.sigmoid(
            self.confidence_scale * normalized + self.confidence_bias
        )
        return gate * seen

    def forward(self, feature, confidence, view_count):
        feature = torch.nan_to_num(
            feature.float(), nan=0.0, posinf=1e6, neginf=-1e6
        )
        if feature.ndim != 3 or feature.shape[-1] != self.feat_dim:
            raise ValueError(
                'c_vggt_feat must have shape [batch, points, '
                f'{self.feat_dim}], got {tuple(feature.shape)}'
            )
        gate = self.confidence_gate(feature, confidence, view_count)
        projected = self.projection(self.norm(feature))
        return projected * gate.unsqueeze(-1)


@Model.register()
class CDMVGGTAdapter(nn.Module):
    """Mask-only PointTransformer plus an optional residual VGGT adapter.

    The PointTransformer and output head are constructed before the optional
    adapter. With a fixed seed their initialization is therefore identical for
    mask-only and VGGT runs, and the zero-initialized adapter preserves the
    mask-only forward pass at step zero.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.use_vggt = bool(cfg.use_vggt)
        self.use_reliability = bool(cfg.use_reliability)
        self.freeze_base = bool(cfg.get('freeze_base', False))
        self.reliability_mode = str(cfg.get('reliability_mode', 'relative'))

        if self.freeze_base and not self.use_vggt:
            raise ValueError('freeze_base=True requires use_vggt=True')

        # Keep this construction order identical to models.cdm.CDM.
        self.scene_model = load_scene_model(
            cfg.scene_model.name,
            cfg.input_feats,
            cfg.scene_model.num_points,
            cfg.scene_model.pretrained_weight,
            freeze=cfg.scene_model.freeze,
        )
        self.contact_layer = nn.Linear(32, 1, bias=True)

        if self.use_vggt:
            self.vggt_adapter = VGGTResidualAdapter(
                feat_dim=cfg.vggt_feat_dim,
                hidden_dim=cfg.adapter_hidden_dim,
                out_dim=32,
                use_reliability=self.use_reliability,
                reliability_mode=self.reliability_mode,
                absolute_confidence_log1p_mean=cfg.get(
                    'absolute_confidence_log1p_mean', 0.0
                ),
                absolute_confidence_log1p_std=cfg.get(
                    'absolute_confidence_log1p_std', 1.0
                ),
                view_count_log1p_mean=cfg.get(
                    'view_count_log1p_mean', 0.0
                ),
                view_count_log1p_std=cfg.get(
                    'view_count_log1p_std', 1.0
                ),
            )
        else:
            self.vggt_adapter = None

        if self.freeze_base:
            # Adapter-only training must preserve both the learned weights and
            # PointTransformer BatchNorm buffers from the initialization
            # checkpoint. ``requires_grad_(False)`` handles the former; the
            # train() override below keeps the frozen modules in eval mode so
            # their running statistics cannot drift.
            self.scene_model.requires_grad_(False)
            self.contact_layer.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.scene_model.eval()
            self.contact_layer.eval()
        return self

    def forward(self, x, **kwargs):
        point_features = self.scene_model((kwargs['c_pc_xyz'], x))
        if self.vggt_adapter is not None:
            point_features = point_features + self.vggt_adapter(
                kwargs['c_vggt_feat'],
                kwargs.get('c_vggt_conf'),
                kwargs.get('c_vggt_view_count'),
            )
        return self.contact_layer(point_features)


@Model.register()
class CDMVGGTAdditiveStem(nn.Module):
    """Reliability-gated VGGT injection before spatial point attention.

    The mask-only input stem is evaluated unchanged. A zero-initialized,
    scalar-gated VGGT projection is added to that stem output before the first
    PointTransformer block, allowing every encoder/decoder stage to reason over
    the dense descriptor while preserving the base checkpoint exactly at step
    zero.
    """

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.use_vggt = True
        self.use_reliability = True
        self.freeze_base = bool(cfg.get('freeze_base', True))
        self.freeze_batchnorm = bool(cfg.get('freeze_batchnorm', False))

        self.scene_model = load_scene_model(
            cfg.scene_model.name,
            cfg.input_feats,
            cfg.scene_model.num_points,
            cfg.scene_model.pretrained_weight,
            freeze=cfg.scene_model.freeze,
        )
        self.contact_layer = nn.Linear(32, 1, bias=True)

        stem = self.scene_model.enc1[0]
        if stem.stride != 1:
            raise ValueError('additive VGGT fusion requires a stride-one input stem')
        stem_dim = int(stem.linear.out_features)
        configured_stem_dim = int(cfg.get('stem_dim', stem_dim))
        if configured_stem_dim != stem_dim:
            raise ValueError(
                f'configured stem_dim {configured_stem_dim} does not match '
                f'PointTransformer stem width {stem_dim}'
            )
        self.vggt_adapter = VGGTAdditiveStemAdapter(
            feat_dim=cfg.vggt_feat_dim,
            stem_dim=stem_dim,
        )

        if self.freeze_base:
            self.scene_model.requires_grad_(False)
            self.contact_layer.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.scene_model.eval()
            self.contact_layer.eval()
            self.vggt_adapter.train(mode)
        elif self.freeze_batchnorm:
            # Joint fine-tuning uses small per-GPU batches. Preserve the base
            # checkpoint's running statistics while allowing the spatial
            # weights and contact head to adapt to the new stem residuals.
            for module in self.scene_model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, x, **kwargs):
        stem_residual = self.vggt_adapter(
            kwargs['c_vggt_feat'],
            kwargs['c_vggt_conf'],
            kwargs['c_vggt_view_count'],
        )
        point_features = self.scene_model(
            (kwargs['c_pc_xyz'], x),
            stem_residual=stem_residual,
        )
        return self.contact_layer(point_features)
