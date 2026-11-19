"""
PKD_CAG_model.py

PKD-CAG model architecture:
- Physiological teacher: dual PPG/GSR streams, 1D-CNN + Mamba sequence encoder, fusion, teacher classifier.
- Video student: ResNet-18 frame encoder + Attention-LSTM temporal encoder, student classifier.
- Conflict-Aware Autonomic Gate (CAG): cosine-disagreement gate and gated classifier.
- Distillation loss: supports both KL soft-label distillation and gated valence-arousal MSE.

Expected input shapes
---------------------
ppg:   Tensor [B, 1, T_ppg] or [B, T_ppg]
gsr:   Tensor [B, 1, T_gsr] or [B, T_gsr]
video: Tensor [B, T_frames, 3, H, W] or [B, 3, H, W]
y:     Tensor [B] with class indices in [0, num_classes - 1]

For exact Mamba use:
    pip install mamba-ssm causal-conv1d
If mamba_ssm is unavailable, the code falls back to a BiGRU sequence encoder unless
ModelConfig(strict_mamba=True) is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet18_Weights, resnet18
except Exception as exc:  # pragma: no cover
    ResNet18_Weights = None
    resnet18 = None
    _TORCHVISION_IMPORT_ERROR = exc
else:
    _TORCHVISION_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba
except Exception:  # pragma: no cover
    Mamba = None


@dataclass
class ModelConfig:
    """Model architecture and optimization settings."""

    num_classes: int = 5

    # Physiological teacher
    ppg_in_channels: int = 1
    gsr_in_channels: int = 1
    cnn_channels: Tuple[int, int, int] = (32, 64, 128)
    physio_stream_dim: int = 128
    physio_fusion_dim: int = 256
    mamba_layers: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    strict_mamba: bool = False

    # Video student
    pretrained_resnet: bool = True
    freeze_resnet: bool = False
    frame_dim: int = 512
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional_lstm: bool = True
    video_dim: int = 256

    # CAG and classifiers
    gate_dim: int = 256
    classifier_hidden_dim: int = 256
    dropout: float = 0.4
    cag_threshold: float = 0.5
    cag_epsilon: float = 0.10

    # Loss defaults from Algorithm 1
    temperature: float = 4.0
    alpha_ce: float = 0.5
    beta_kd: float = 0.5


class Conv1DBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU + MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PhysioCNNEncoder(nn.Module):
    """Three-block 1D-CNN encoder for PPG or GSR local morphology."""

    def __init__(self, in_channels: int = 1, channels: Tuple[int, int, int] = (32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            Conv1DBlock(in_channels, c1, kernel_size=3),
            Conv1DBlock(c1, c2, kernel_size=3),
            Conv1DBlock(c2, c3, kernel_size=3),
        )
        self.out_dim = c3

    def forward(self, x: Tensor) -> Tensor:
        # Accept [B, T] and convert to [B, 1, T]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Physiological input must be [B, T] or [B, C, T], got {tuple(x.shape)}")
        return self.net(x)  # [B, C, T']


class ResidualMambaBlock(nn.Module):
    """LayerNorm -> Mamba -> residual block for [B, L, D] sequences."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed. Install mamba-ssm or set strict_mamba=False.")
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mamba(self.norm(x))


class MambaSequenceEncoder(nn.Module):
    """
    Sequence encoder used after the 1D-CNN.

    Exact path: stacked Mamba blocks.
    Fallback path: BiGRU with projection to the same dimension, for environments where mamba_ssm
    is unavailable. Use strict_mamba=True if you do not want the fallback.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        strict_mamba: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.using_mamba = Mamba is not None

        if self.using_mamba:
            self.encoder = nn.ModuleList(
                [ResidualMambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
            )
            self.fallback = None
        else:
            if strict_mamba:
                raise ImportError("mamba_ssm is required for strict Mamba mode. Install mamba-ssm causal-conv1d.")
            warnings.warn(
                "mamba_ssm is not installed; using a BiGRU fallback. "
                "Install mamba-ssm to enable the Mamba encoder.",
                RuntimeWarning,
            )
            hidden = max(d_model // 2, 1)
            self.fallback = nn.GRU(
                input_size=d_model,
                hidden_size=hidden,
                num_layers=max(num_layers, 1),
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fallback_proj = nn.Linear(hidden * 2, d_model)
            self.encoder = nn.ModuleList()

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, L, D]
        if self.using_mamba:
            for layer in self.encoder:
                x = layer(x)
        else:
            x, _ = self.fallback(x)
            x = self.fallback_proj(x)
        return self.final_norm(x)


class PhysioStreamEncoder(nn.Module):
    """One physiological stream: 1D-CNN -> Mamba/BiGRU -> temporal pooling -> projection."""

    def __init__(self, in_channels: int, cfg: ModelConfig):
        super().__init__()
        self.cnn = PhysioCNNEncoder(in_channels=in_channels, channels=cfg.cnn_channels)
        cnn_dim = cfg.cnn_channels[-1]
        if cnn_dim != cfg.physio_stream_dim:
            self.input_proj = nn.Linear(cnn_dim, cfg.physio_stream_dim)
        else:
            self.input_proj = nn.Identity()
        self.sequence_encoder = MambaSequenceEncoder(
            d_model=cfg.physio_stream_dim,
            num_layers=cfg.mamba_layers,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            strict_mamba=cfg.strict_mamba,
            dropout=cfg.dropout,
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.physio_stream_dim, cfg.physio_stream_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.physio_stream_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)               # [B, C, L]
        x = x.transpose(1, 2)         # [B, L, C]
        x = self.input_proj(x)        # [B, L, D]
        x = self.sequence_encoder(x)  # [B, L, D]
        x = x.mean(dim=1)             # [B, D]
        return self.out(x)            # [B, D]


class MLPClassifier(nn.Module):
    """Two-layer MLP classifier head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhysiologicalTeacher(nn.Module):
    """Dual-stream PPG/GSR teacher network."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.ppg_encoder = PhysioStreamEncoder(cfg.ppg_in_channels, cfg)
        self.gsr_encoder = PhysioStreamEncoder(cfg.gsr_in_channels, cfg)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.physio_stream_dim * 2, cfg.physio_fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.physio_fusion_dim),
        )
        self.classifier = MLPClassifier(
            cfg.physio_fusion_dim,
            cfg.classifier_hidden_dim,
            cfg.num_classes,
            dropout=cfg.dropout,
        )
        # Optional dimensional affect head for Section 3.3.4 style VA distillation.
        self.va_head = nn.Sequential(
            nn.Linear(cfg.physio_fusion_dim, cfg.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden_dim, 2),
        )

    def forward(self, ppg: Tensor, gsr: Tensor, temperature: Optional[float] = None) -> Dict[str, Tensor]:
        f_ppg = self.ppg_encoder(ppg)              # [B, 128]
        f_gsr = self.gsr_encoder(gsr)              # [B, 128]
        f_physio = self.fusion(torch.cat([f_ppg, f_gsr], dim=-1))  # [B, 256]
        logits = self.classifier(f_physio)          # [B, C]
        probs = F.softmax(logits, dim=-1)
        soft_probs = F.softmax(logits / temperature, dim=-1) if temperature is not None else probs
        va = self.va_head(f_physio)                 # [B, 2]
        return {
            "f_ppg": f_ppg,
            "f_gsr": f_gsr,
            "features": f_physio,
            "logits": logits,
            "probs": probs,
            "soft_probs": soft_probs,
            "va": va,
        }


class ResNet18FrameEncoder(nn.Module):
    """Frame-level spatial encoder producing 512-D embeddings per frame."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        if resnet18 is None:
            raise ImportError(f"torchvision could not be imported: {_TORCHVISION_IMPORT_ERROR}")

        weights = None
        if pretrained:
            if ResNet18_Weights is None:
                weights = None
            else:
                weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])  # remove fc, keep avgpool
        self.out_dim = 512

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, video: Tensor) -> Tensor:
        # Accept [B, 3, H, W] as a single-frame sequence.
        if video.ndim == 4:
            video = video.unsqueeze(1)
        if video.ndim != 5:
            raise ValueError(f"Video input must be [B, T, 3, H, W] or [B, 3, H, W], got {tuple(video.shape)}")

        b, t, c, h, w = video.shape
        x = video.reshape(b * t, c, h, w)
        x = self.backbone(x).flatten(1)  # [B*T, 512]
        return x.reshape(b, t, self.out_dim)


class AttentionLSTM(nn.Module):
    """Attention-weighted LSTM temporal aggregation."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.out_dim = out_dim
        self.attention = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, 1, bias=False),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, seq: Tensor, lengths: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # seq: [B, T, D]
        h, _ = self.lstm(seq)  # [B, T, H]
        scores = self.attention(h).squeeze(-1)  # [B, T]

        if lengths is not None:
            max_t = seq.size(1)
            mask = torch.arange(max_t, device=seq.device).unsqueeze(0) >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=1)  # [B, T]
        context = torch.sum(h * attn.unsqueeze(-1), dim=1)  # [B, H]
        return self.norm(context), attn


class VideoStudent(nn.Module):
    """ResNet-18 + Attention-LSTM video student network."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = ResNet18FrameEncoder(pretrained=cfg.pretrained_resnet, freeze=cfg.freeze_resnet)
        self.temporal = AttentionLSTM(
            input_dim=cfg.frame_dim,
            hidden_dim=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            bidirectional=cfg.bidirectional_lstm,
            dropout=cfg.dropout,
        )
        temporal_dim = self.temporal.out_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(temporal_dim, cfg.video_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.LayerNorm(cfg.video_dim),
        )
        self.classifier = MLPClassifier(
            cfg.video_dim,
            cfg.classifier_hidden_dim,
            cfg.num_classes,
            dropout=cfg.dropout,
        )
        # Optional dimensional affect head for VA distillation.
        self.va_head = nn.Sequential(
            nn.Linear(cfg.video_dim, cfg.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden_dim, 2),
        )

    def forward(self, video: Tensor, lengths: Optional[Tensor] = None, temperature: Optional[float] = None) -> Dict[str, Tensor]:
        frame_seq = self.frame_encoder(video)          # [B, T, 512]
        context, attn = self.temporal(frame_seq, lengths=lengths)  # [B, 256], [B, T]
        f_video = self.feature_proj(context)           # [B, 256]
        logits = self.classifier(f_video)              # [B, C]
        probs = F.softmax(logits, dim=-1)
        soft_probs = F.softmax(logits / temperature, dim=-1) if temperature is not None else probs
        va = self.va_head(f_video)                     # [B, 2]
        return {
            "frame_features": frame_seq,
            "attention": attn,
            "features": f_video,
            "logits": logits,
            "probs": probs,
            "soft_probs": soft_probs,
            "va": va,
        }


class ConflictAwareAutonomicGate(nn.Module):
    """
    Conflict-Aware Autonomic Gate.

    Computes:
        delta = 1 - cosine(z_video, z_physio)
        omega = sigmoid(gamma * delta + beta_g)
        z_fused = (1 - omega) * z_video + omega * z_physio
    """

    def __init__(
        self,
        video_dim: int = 256,
        physio_dim: int = 256,
        gate_dim: int = 256,
        threshold: float = 0.5,
        epsilon: float = 0.10,
    ):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, gate_dim)
        self.physio_proj = nn.Linear(physio_dim, gate_dim)
        # Positive gamma via softplus. beta_g is unrestricted.
        self.log_gamma = nn.Parameter(torch.tensor(1.0))
        self.beta_g = nn.Parameter(torch.tensor(0.0))
        self.threshold = threshold
        self.epsilon = epsilon

    @property
    def gamma(self) -> Tensor:
        return F.softplus(self.log_gamma)

    def project_video(self, f_video: Tensor) -> Tensor:
        return self.video_proj(f_video)

    def project_physio(self, f_physio: Tensor) -> Tensor:
        return self.physio_proj(f_physio)

    def forward(self, f_video: Tensor, f_physio: Tensor, amplify_high_conflict: bool = True) -> Dict[str, Tensor]:
        z_video = self.project_video(f_video)
        z_physio = self.project_physio(f_physio)

        cosine = F.cosine_similarity(z_video, z_physio, dim=-1, eps=1e-8)
        delta = (1.0 - cosine).clamp(min=0.0, max=2.0)  # [B]

        omega_physio = torch.sigmoid(self.gamma * delta + self.beta_g)  # [B]
        if amplify_high_conflict and self.epsilon > 0:
            omega_physio = torch.where(
                delta > self.threshold,
                torch.clamp(omega_physio + self.epsilon, max=1.0),
                omega_physio,
            )
        omega_video = 1.0 - omega_physio

        z_fused = omega_video.unsqueeze(-1) * z_video + omega_physio.unsqueeze(-1) * z_physio
        return {
            "z_video": z_video,
            "z_physio": z_physio,
            "z_fused": z_fused,
            "conflict_delta": delta,
            "omega_physio": omega_physio,
            "omega_video": omega_video,
            "cosine": cosine,
        }


class PKDCAG(nn.Module):
    """Full teacher-student PKD-CAG model."""

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.teacher = PhysiologicalTeacher(self.cfg)
        self.student = VideoStudent(self.cfg)
        self.cag = ConflictAwareAutonomicGate(
            video_dim=self.cfg.video_dim,
            physio_dim=self.cfg.physio_fusion_dim,
            gate_dim=self.cfg.gate_dim,
            threshold=self.cfg.cag_threshold,
            epsilon=self.cfg.cag_epsilon,
        )
        self.gated_classifier = MLPClassifier(
            self.cfg.gate_dim,
            self.cfg.classifier_hidden_dim,
            self.cfg.num_classes,
            dropout=self.cfg.dropout,
        )

    def forward(
        self,
        video: Tensor,
        ppg: Optional[Tensor] = None,
        gsr: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        teacher_detach: bool = True,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        During training with physiological signals, pass video, ppg, and gsr.
        During deployment/inference, pass only video; the student branch is used.
        """
        if temperature is None:
            temperature = self.cfg.temperature

        student_out = self.student(video, lengths=lengths, temperature=temperature)
        out: Dict[str, Tensor] = {
            "student_features": student_out["features"],
            "student_logits": student_out["logits"],
            "student_probs": student_out["probs"],
            "student_soft_probs": student_out["soft_probs"],
            "student_va": student_out["va"],
            "student_attention": student_out["attention"],
        }

        # Deployment path: no PPG/GSR. Inference uses student logits.
        z_video = self.cag.project_video(student_out["features"])
        out["visual_projected_logits"] = self.gated_classifier(z_video)
        out["deployment_logits"] = student_out["logits"]
        out["deployment_probs"] = student_out["probs"]

        if ppg is None or gsr is None:
            return out

        teacher_out = self.teacher(ppg, gsr, temperature=temperature)
        f_physio = teacher_out["features"].detach() if teacher_detach else teacher_out["features"]
        teacher_logits = teacher_out["logits"].detach() if teacher_detach else teacher_out["logits"]
        teacher_soft_probs = teacher_out["soft_probs"].detach() if teacher_detach else teacher_out["soft_probs"]
        teacher_va = teacher_out["va"].detach() if teacher_detach else teacher_out["va"]

        gate_out = self.cag(student_out["features"], f_physio)
        fused_logits = self.gated_classifier(gate_out["z_fused"])
        fused_probs = F.softmax(fused_logits, dim=-1)

        out.update(
            {
                "teacher_features": f_physio,
                "teacher_logits": teacher_logits,
                "teacher_soft_probs": teacher_soft_probs,
                "teacher_va": teacher_va,
                "fused_logits": fused_logits,
                "fused_probs": fused_probs,
                **gate_out,
            }
        )
        return out

    @torch.no_grad()
    def predict(self, video: Tensor, lengths: Optional[Tensor] = None, use_projected_classifier: bool = False) -> Tensor:
        """Video-only prediction for deployment."""
        self.eval()
        out = self.forward(video=video, lengths=lengths, ppg=None, gsr=None)
        logits = out["visual_projected_logits"] if use_projected_classifier else out["student_logits"]
        return torch.argmax(logits, dim=-1)


class PKDCAGLoss(nn.Module):
    """
    Loss module for PKD-CAG.

    ce_on:
        "student" -> CE on student logits.
        "fused"   -> CE on CAG fused logits.
        "both"    -> average CE on student and fused logits.

    kd_type:
        "kl"     -> KL soft-label distillation (alternative objective).
        "va_mse" -> Gated valence-arousal squared-error distillation.
        "both"   -> sum of KL and gated VA MSE.
    """

    def __init__(
        self,
        alpha_ce: float = 0.5,
        beta_kd: float = 0.5,
        temperature: float = 4.0,
        ce_on: Literal["student", "fused", "both"] = "both",
        kd_type: Literal["kl", "va_mse", "both"] = "kl",
    ):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.beta_kd = beta_kd
        self.temperature = temperature
        self.ce_on = ce_on
        self.kd_type = kd_type

    def _ce_loss(self, outputs: Dict[str, Tensor], y: Tensor) -> Tensor:
        student_ce = F.cross_entropy(outputs["student_logits"], y)
        fused_logits = outputs.get("fused_logits")

        if self.ce_on == "student" or fused_logits is None:
            return student_ce
        fused_ce = F.cross_entropy(fused_logits, y)
        if self.ce_on == "fused":
            return fused_ce
        return 0.5 * (student_ce + fused_ce)

    def _kl_kd_loss(self, outputs: Dict[str, Tensor]) -> Tensor:
        if "teacher_logits" not in outputs:
            return torch.tensor(0.0, device=outputs["student_logits"].device)
        t = self.temperature
        student_log_probs = F.log_softmax(outputs["student_logits"] / t, dim=-1)
        teacher_probs = F.softmax(outputs["teacher_logits"] / t, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)

    def _va_mse_loss(self, outputs: Dict[str, Tensor]) -> Tensor:
        if "teacher_va" not in outputs:
            return torch.tensor(0.0, device=outputs["student_logits"].device)
        per_sample = F.mse_loss(outputs["student_va"], outputs["teacher_va"], reduction="none").sum(dim=-1)
        omega = outputs.get("omega_physio")
        if omega is not None:
            per_sample = omega.detach() * per_sample
        return per_sample.mean()

    def forward(self, outputs: Dict[str, Tensor], y: Tensor) -> Dict[str, Tensor]:
        ce = self._ce_loss(outputs, y)

        if self.kd_type == "kl":
            kd = self._kl_kd_loss(outputs)
        elif self.kd_type == "va_mse":
            kd = self._va_mse_loss(outputs)
        elif self.kd_type == "both":
            kd = self._kl_kd_loss(outputs) + self._va_mse_loss(outputs)
        else:  # pragma: no cover
            raise ValueError(f"Unknown kd_type={self.kd_type}")

        total = self.alpha_ce * ce + self.beta_kd * kd
        return {"loss": total, "ce_loss": ce.detach(), "kd_loss": kd.detach()}


def freeze_module(module: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = not freeze


def build_pkd_cag(num_classes: int = 5, strict_mamba: bool = False, pretrained_resnet: bool = True) -> PKDCAG:
    """Convenience builder."""
    cfg = ModelConfig(num_classes=num_classes, strict_mamba=strict_mamba, pretrained_resnet=pretrained_resnet)
    return PKDCAG(cfg)


def teacher_training_step(
    model: PKDCAG,
    ppg: Tensor,
    gsr: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One supervised teacher step using PPG/GSR only."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model.teacher(ppg, gsr)
    loss = F.cross_entropy(out["logits"], y)
    loss.backward()
    optimizer.step()
    return {"teacher_loss": float(loss.detach().cpu())}


def student_distillation_step(
    model: PKDCAG,
    video: Tensor,
    ppg: Tensor,
    gsr: Tensor,
    y: Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[PKDCAGLoss] = None,
    freeze_teacher: bool = True,
) -> Dict[str, float]:
    """One PKD-CAG student distillation step with the physiological teacher."""
    if criterion is None:
        criterion = PKDCAGLoss(
            alpha_ce=model.cfg.alpha_ce,
            beta_kd=model.cfg.beta_kd,
            temperature=model.cfg.temperature,
            ce_on="both",
            kd_type="kl",
        )

    model.train()
    if freeze_teacher:
        model.teacher.eval()
        freeze_module(model.teacher, True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(video=video, ppg=ppg, gsr=gsr, teacher_detach=freeze_teacher, temperature=criterion.temperature)
    losses = criterion(outputs, y)
    losses["loss"].backward()
    optimizer.step()

    metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
    if "omega_physio" in outputs:
        metrics["omega_physio_mean"] = float(outputs["omega_physio"].detach().mean().cpu())
        metrics["conflict_delta_mean"] = float(outputs["conflict_delta"].detach().mean().cpu())
    return metrics


if __name__ == "__main__":
    # Smoke test with small tensors. Use pretrained_resnet=False to avoid downloading weights.
    cfg = ModelConfig(pretrained_resnet=False, strict_mamba=False)
    model = PKDCAG(cfg)

    B = 2
    video = torch.randn(B, 4, 3, 224, 224)
    ppg = torch.randn(B, 1, 1500)  # e.g., 60 s at 25 Hz
    gsr = torch.randn(B, 1, 900)   # e.g., 60 s at 15 Hz
    y = torch.randint(0, cfg.num_classes, (B,))

    out = model(video=video, ppg=ppg, gsr=gsr)
    crit = PKDCAGLoss(alpha_ce=0.5, beta_kd=0.5, temperature=4.0, ce_on="both", kd_type="kl")
    loss_dict = crit(out, y)
    print("student_logits:", tuple(out["student_logits"].shape))
    print("teacher_logits:", tuple(out["teacher_logits"].shape))
    print("fused_logits:", tuple(out["fused_logits"].shape))
    print("loss:", float(loss_dict["loss"].detach()))
