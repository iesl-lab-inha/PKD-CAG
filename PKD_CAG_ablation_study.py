"""
PKD-CAG ablation-study
=============================

1. Video student only (baseline)
2. PPG teacher + physiological KD, no CAG
3. GSR teacher + physiological KD, no CAG
4. PPG+GSR teacher + physiological KD, no CAG (fixed KD weight)
5. Full PKD-CAG (PPG+GSR KD + Conflict-Aware Autonomic Gate)

Important scientific/reproducibility notes
------------------------------------------
* Test metrics are always computed from the VIDEO STUDENT ONLY, matching the
  manuscript's biosensor-free deployment setting.
* The default KD mode is valence-arousal MSE, as defined in the paper.
  Therefore, batches should contain a two-dimensional ``va`` target when
  teacher VA heads are being trained.
* The fixed-weight ablations use the same alpha/beta loss weights but DO NOT
  use conflict-aware sample weighting.

Expected batch format
---------------------
Preferred: dictionary with
    {
        "video": Tensor [B,T,3,H,W] or [B,3,H,W],
        "ppg":   Tensor [B,1,T_ppg] or [B,T_ppg],
        "gsr":   Tensor [B,1,T_gsr] or [B,T_gsr],
        "label": Tensor [B],                 # aliases: y, labels, target
        "va":    Tensor [B,2],               # optional for baseline, required
                                               # for valence-arousal distillation
        "lengths": Tensor [B],               # optional video lengths
    }

A tuple/list ``(video, ppg, gsr, label)`` or
``(video, ppg, gsr, label, va)`` is also accepted.

Example
-------
    from PKD_CAG_ablation_study import AblationTrainConfig, run_all_ablations

    cfg = AblationTrainConfig(output_dir="ablation_results")
    results = run_all_ablations(train_loader, val_loader, test_loader, cfg)

The function writes:
    ablation_results.csv
    manuscript_reference_comparison.csv
    checkpoints/*.pt
    cag_behavior.csv             # for the full PKD-CAG variant

Dataset-specific preprocessing and DataLoaders are external to this module.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from PKD_CAG_model import (
    ConflictAwareAutonomicGate,
    MLPClassifier,
    ModelConfig,
    PhysioStreamEncoder,
    PhysiologicalTeacher,
    VideoStudent,
    freeze_module,
)


VariantName = Literal[
    "video_only",
    "ppg_kd",
    "gsr_kd",
    "ppg_gsr_kd_no_cag",
    "pkd_cag",
]


VARIANT_ORDER: Tuple[VariantName, ...] = (
    "video_only",
    "ppg_kd",
    "gsr_kd",
    "ppg_gsr_kd_no_cag",
    "pkd_cag",
)


VARIANT_DISPLAY_NAMES: Dict[VariantName, str] = {
    "video_only": "Video student only (baseline)",
    "ppg_kd": "PPG teacher + KD",
    "gsr_kd": "GSR teacher + KD",
    "ppg_gsr_kd_no_cag": "PPG+GSR teacher + KD (PKD w/o CAG)",
    "pkd_cag": "PKD-CAG (proposed)",
}


# Reference values reported in Table IV of the paper.
# These values are NEVER used by the training/evaluation code.
MANUSCRIPT_REFERENCE_RESULTS: Dict[VariantName, Dict[str, float]] = {
    "video_only": {"accuracy": 85.79, "f1": 84.62},
    "ppg_kd": {"accuracy": 85.91, "f1": 85.23},
    "gsr_kd": {"accuracy": 87.89, "f1": 85.93},
    "ppg_gsr_kd_no_cag": {"accuracy": 90.02, "f1": 89.83},
    "pkd_cag": {"accuracy": 92.29, "f1": 92.01},
}


@dataclass
class AblationTrainConfig:
    # Manuscript Table III defaults
    max_epochs: int = 100
    teacher_epochs: Optional[int] = None  # None -> use max_epochs
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    patience: int = 15
    alpha_ce: float = 0.5
    beta_kd: float = 0.5
    temperature: float = 4.0
    conflict_threshold: float = 0.5

    # Teacher pretraining.
    # The discrete CE term learns emotion discrimination; the VA term trains
    # the teacher's VA head used as the physiological distillation target.
    teacher_ce_weight: float = 1.0
    teacher_va_weight: float = 1.0

    # Main experiment settings
    num_classes: int = 5
    pretrained_resnet: bool = True
    strict_mamba: bool = True
    seed: int = 42
    grad_clip_norm: Optional[float] = 5.0
    device: Optional[str] = None
    output_dir: str = "ablation_results"

    # Full CAG training uses a CAG-weighted
    # VA loss plus a fused CE term so gate parameters get a supervised signal.
    cag_ce_on_both: bool = True
    detach_cag_weight_in_kd: bool = True

    # If True, KD variants require VA labels while pretraining the teacher.
    # VA supervision is enabled by default.
    require_va_targets: bool = True


@dataclass(frozen=True)
class VariantSpec:
    name: VariantName
    teacher_mode: Literal["none", "ppg", "gsr", "both"]
    use_kd: bool
    use_cag: bool

    @property
    def use_ppg(self) -> bool:
        return self.teacher_mode in ("ppg", "both")

    @property
    def use_gsr(self) -> bool:
        return self.teacher_mode in ("gsr", "both")


VARIANTS: Dict[VariantName, VariantSpec] = {
    "video_only": VariantSpec("video_only", "none", False, False),
    "ppg_kd": VariantSpec("ppg_kd", "ppg", True, False),
    "gsr_kd": VariantSpec("gsr_kd", "gsr", True, False),
    "ppg_gsr_kd_no_cag": VariantSpec("ppg_gsr_kd_no_cag", "both", True, False),
    "pkd_cag": VariantSpec("pkd_cag", "both", True, True),
}


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n: int


@dataclass
class AblationResult:
    variant: str
    ppg: bool
    gsr: bool
    video: bool
    kd: bool
    cag: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_test: int
    best_epoch: int
    best_val_loss: float
    checkpoint: str


class SingleModalityTeacher(nn.Module):
    """PPG-only or GSR-only teacher with the same physiological backbone family."""

    def __init__(self, cfg: ModelConfig, modality: Literal["ppg", "gsr"]):
        super().__init__()
        self.cfg = cfg
        self.modality = modality
        in_channels = cfg.ppg_in_channels if modality == "ppg" else cfg.gsr_in_channels
        self.encoder = PhysioStreamEncoder(in_channels, cfg)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.physio_stream_dim, cfg.physio_fusion_dim),
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
        self.va_head = nn.Sequential(
            nn.Linear(cfg.physio_fusion_dim, cfg.classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden_dim, 2),
        )

    def forward(self, x: Tensor, temperature: Optional[float] = None) -> Dict[str, Tensor]:
        f_stream = self.encoder(x)
        f_physio = self.fusion(f_stream)
        logits = self.classifier(f_physio)
        probs = F.softmax(logits, dim=-1)
        if temperature is None:
            soft_probs = probs
        else:
            soft_probs = F.softmax(logits / temperature, dim=-1)
        va = self.va_head(f_physio)
        return {
            "features": f_physio,
            "logits": logits,
            "probs": probs,
            "soft_probs": soft_probs,
            "va": va,
        }


class AblationModel(nn.Module):
    """One controlled PKD-CAG ablation configuration."""

    def __init__(self, model_cfg: ModelConfig, spec: VariantSpec):
        super().__init__()
        self.cfg = model_cfg
        self.spec = spec
        self.student = VideoStudent(model_cfg)

        if spec.teacher_mode == "none":
            self.teacher: Optional[nn.Module] = None
        elif spec.teacher_mode == "ppg":
            self.teacher = SingleModalityTeacher(model_cfg, "ppg")
        elif spec.teacher_mode == "gsr":
            self.teacher = SingleModalityTeacher(model_cfg, "gsr")
        elif spec.teacher_mode == "both":
            self.teacher = PhysiologicalTeacher(model_cfg)
        else:  # pragma: no cover
            raise ValueError(f"Unknown teacher mode: {spec.teacher_mode}")

        if spec.use_cag:
            self.cag: Optional[ConflictAwareAutonomicGate] = ConflictAwareAutonomicGate(
                video_dim=model_cfg.video_dim,
                physio_dim=model_cfg.physio_fusion_dim,
                gate_dim=model_cfg.gate_dim,
                threshold=model_cfg.cag_threshold,
                epsilon=model_cfg.cag_epsilon,
            )
            self.gated_classifier: Optional[MLPClassifier] = MLPClassifier(
                model_cfg.gate_dim,
                model_cfg.classifier_hidden_dim,
                model_cfg.num_classes,
                dropout=model_cfg.dropout,
            )
        else:
            self.cag = None
            self.gated_classifier = None

    def _teacher_forward(
        self,
        ppg: Optional[Tensor],
        gsr: Optional[Tensor],
        temperature: float,
    ) -> Dict[str, Tensor]:
        if self.teacher is None:
            raise RuntimeError("This ablation variant has no physiological teacher.")
        if self.spec.teacher_mode == "ppg":
            if ppg is None:
                raise ValueError("PPG tensor is required for the PPG-teacher ablation.")
            return self.teacher(ppg, temperature=temperature)  # type: ignore[misc]
        if self.spec.teacher_mode == "gsr":
            if gsr is None:
                raise ValueError("GSR tensor is required for the GSR-teacher ablation.")
            return self.teacher(gsr, temperature=temperature)  # type: ignore[misc]
        if ppg is None or gsr is None:
            raise ValueError("Both PPG and GSR are required for the dual-teacher ablation.")
        return self.teacher(ppg, gsr, temperature=temperature)  # type: ignore[misc]

    def forward(
        self,
        video: Tensor,
        ppg: Optional[Tensor] = None,
        gsr: Optional[Tensor] = None,
        lengths: Optional[Tensor] = None,
        temperature: float = 4.0,
        teacher_detach: bool = True,
    ) -> Dict[str, Tensor]:
        student_out = self.student(video, lengths=lengths, temperature=temperature)
        out: Dict[str, Tensor] = {
            "student_features": student_out["features"],
            "student_logits": student_out["logits"],
            "student_probs": student_out["probs"],
            "student_soft_probs": student_out["soft_probs"],
            "student_va": student_out["va"],
            "student_attention": student_out["attention"],
        }

        if self.teacher is None:
            return out

        teacher_out = self._teacher_forward(ppg, gsr, temperature)
        for key, dst_key in (
            ("features", "teacher_features"),
            ("logits", "teacher_logits"),
            ("soft_probs", "teacher_soft_probs"),
            ("va", "teacher_va"),
        ):
            value = teacher_out[key]
            out[dst_key] = value.detach() if teacher_detach else value

        if self.spec.use_cag:
            assert self.cag is not None and self.gated_classifier is not None
            gate_out = self.cag(out["student_features"], out["teacher_features"])
            fused_logits = self.gated_classifier(gate_out["z_fused"])
            out["fused_logits"] = fused_logits
            out["fused_probs"] = F.softmax(fused_logits, dim=-1)
            out.update(gate_out)

        return out

    @torch.no_grad()
    def predict_video_only(self, video: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """Deployment prediction: physiology/CAG are not needed at inference."""
        self.eval()
        out = self.student(video, lengths=lengths)
        return torch.argmax(out["logits"], dim=-1)


# ---------------------------------------------------------------------------
# Batch handling
# ---------------------------------------------------------------------------

LABEL_ALIASES = ("label", "labels", "y", "target", "targets")
VA_ALIASES = ("va", "valence_arousal", "va_target", "va_targets")


def _first_present(batch: Mapping[str, object], names: Sequence[str]) -> Optional[object]:
    for name in names:
        if name in batch:
            return batch[name]
    return None


def unpack_batch(batch: object) -> Dict[str, Optional[Tensor]]:
    """Normalize supported batch formats to a common dictionary."""
    if isinstance(batch, Mapping):
        label = _first_present(batch, LABEL_ALIASES)
        if label is None:
            raise KeyError(f"Batch dictionary must contain one of label keys: {LABEL_ALIASES}")
        va = _first_present(batch, VA_ALIASES)
        return {
            "video": batch.get("video"),  # type: ignore[dict-item]
            "ppg": batch.get("ppg"),      # type: ignore[dict-item]
            "gsr": batch.get("gsr"),      # type: ignore[dict-item]
            "label": label,                # type: ignore[dict-item]
            "va": va,                      # type: ignore[dict-item]
            "lengths": batch.get("lengths"),  # type: ignore[dict-item]
        }

    if isinstance(batch, (tuple, list)):
        if len(batch) == 4:
            video, ppg, gsr, label = batch
            va = None
            lengths = None
        elif len(batch) == 5:
            video, ppg, gsr, label, va = batch
            lengths = None
        elif len(batch) == 6:
            video, ppg, gsr, label, va, lengths = batch
        else:
            raise ValueError(
                "Tuple/list batches must be (video, ppg, gsr, label), "
                "(video, ppg, gsr, label, va), or "
                "(video, ppg, gsr, label, va, lengths)."
            )
        return {
            "video": video, "ppg": ppg, "gsr": gsr, "label": label,
            "va": va, "lengths": lengths,
        }

    raise TypeError("Unsupported batch type. Use a dict, tuple, or list.")


def batch_to_device(batch: object, device: torch.device) -> Dict[str, Optional[Tensor]]:
    data = unpack_batch(batch)
    moved: Dict[str, Optional[Tensor]] = {}
    for key, value in data.items():
        if value is None:
            moved[key] = None
        elif torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = torch.as_tensor(value, device=device)
    if moved["video"] is None or moved["label"] is None:
        raise ValueError("Every batch must include video and label tensors.")
    moved["label"] = moved["label"].long()  # type: ignore[union-attr]
    if moved["va"] is not None:
        moved["va"] = moved["va"].float()  # type: ignore[union-attr]
    return moved


# ---------------------------------------------------------------------------
# Reproducibility and metrics
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode can reduce performance and may reject a few operations;
    # use deterministic algorithms where available without making them mandatory.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confusion_matrix_from_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_predictions(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> Metrics:
    cm = confusion_matrix_from_predictions(y_true, y_pred, num_classes)
    n = int(cm.sum())
    accuracy = float(np.trace(cm) / n) if n else 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return Metrics(
        accuracy=100.0 * accuracy,
        precision=100.0 * float(np.mean(precisions)),
        recall=100.0 * float(np.mean(recalls)),
        f1=100.0 * float(np.mean(f1s)),
        n=n,
    )


# ---------------------------------------------------------------------------
# Losses and training
# ---------------------------------------------------------------------------


def teacher_supervision_loss(
    teacher_out: Dict[str, Tensor],
    y: Tensor,
    va_target: Optional[Tensor],
    cfg: AblationTrainConfig,
) -> Tensor:
    ce = F.cross_entropy(teacher_out["logits"], y)
    if va_target is None:
        if cfg.require_va_targets:
            raise ValueError(
                "VA targets are required for physiological VA distillation. "
                "Add batch['va'] with shape [B,2], or set require_va_targets=False "
                "(disables supervised training of the teacher VA head)."
            )
        return cfg.teacher_ce_weight * ce
    va_loss = F.mse_loss(teacher_out["va"], va_target)
    return cfg.teacher_ce_weight * ce + cfg.teacher_va_weight * va_loss


def student_ablation_loss(
    outputs: Dict[str, Tensor],
    y: Tensor,
    spec: VariantSpec,
    cfg: AblationTrainConfig,
) -> Tuple[Tensor, Dict[str, float]]:
    student_ce = F.cross_entropy(outputs["student_logits"], y)

    if not spec.use_kd:
        return student_ce, {
            "loss": float(student_ce.detach().cpu()),
            "ce": float(student_ce.detach().cpu()),
            "kd": 0.0,
        }

    teacher_va = outputs.get("teacher_va")
    if teacher_va is None:
        raise RuntimeError("KD variant did not produce teacher_va.")

    per_sample_kd = F.mse_loss(
        outputs["student_va"], teacher_va, reduction="none"
    ).mean(dim=-1)

    if spec.use_cag:
        omega = outputs.get("omega_physio")
        if omega is None:
            raise RuntimeError("CAG variant did not produce omega_physio.")
        kd_weight = omega.detach() if cfg.detach_cag_weight_in_kd else omega
        kd = (kd_weight * per_sample_kd).mean()

        if cfg.cag_ce_on_both:
            fused_logits = outputs.get("fused_logits")
            if fused_logits is None:
                raise RuntimeError("CAG variant did not produce fused_logits.")
            fused_ce = F.cross_entropy(fused_logits, y)
            ce = 0.5 * (student_ce + fused_ce)
        else:
            ce = student_ce
    else:
        # Controlled fixed-weight KD: no sample-wise conflict weighting.
        kd = per_sample_kd.mean()
        ce = student_ce

    total = cfg.alpha_ce * ce + cfg.beta_kd * kd
    log = {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "kd": float(kd.detach().cpu()),
    }
    if "omega_physio" in outputs:
        log["omega"] = float(outputs["omega_physio"].detach().mean().cpu())
        log["conflict"] = float(outputs["conflict_delta"].detach().mean().cpu())
    return total, log


def _forward_teacher_from_batch(
    model: AblationModel,
    data: Dict[str, Optional[Tensor]],
    temperature: float,
) -> Dict[str, Tensor]:
    if model.teacher is None:
        raise RuntimeError("No teacher in this variant.")
    return model._teacher_forward(data["ppg"], data["gsr"], temperature)


def train_teacher(
    model: AblationModel,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
    cfg: AblationTrainConfig,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    """Pretrain physiological teacher before freezing it for student KD."""
    if model.teacher is None:
        return

    freeze_module(model.teacher, False)
    model.teacher.train()
    epochs = cfg.teacher_epochs if cfg.teacher_epochs is not None else cfg.max_epochs
    optimizer = torch.optim.Adam(
        model.teacher.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=cfg.min_learning_rate
    )

    best_val = math.inf
    bad_epochs = 0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.teacher.train()
        for raw_batch in train_loader:
            data = batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            teacher_out = _forward_teacher_from_batch(model, data, cfg.temperature)
            loss = teacher_supervision_loss(
                teacher_out,
                data["label"],  # type: ignore[arg-type]
                data["va"],
                cfg,
            )
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.teacher.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        scheduler.step()

        model.teacher.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for raw_batch in val_loader:
                data = batch_to_device(raw_batch, device)
                out = _forward_teacher_from_batch(model, data, cfg.temperature)
                loss = teacher_supervision_loss(
                    out,
                    data["label"],  # type: ignore[arg-type]
                    data["va"],
                    cfg,
                )
                b = int(data["label"].shape[0])  # type: ignore[union-attr]
                val_sum += float(loss.cpu()) * b
                val_n += b
        val_loss = val_sum / max(val_n, 1)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(model.teacher.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.teacher.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.teacher.state_dict(), checkpoint_path)
    model.teacher.eval()
    freeze_module(model.teacher, True)


def _train_one_student_epoch(
    model: AblationModel,
    loader: Iterable[object],
    optimizer: torch.optim.Optimizer,
    spec: VariantSpec,
    cfg: AblationTrainConfig,
    device: torch.device,
) -> float:
    model.train()
    if model.teacher is not None:
        model.teacher.eval()
        freeze_module(model.teacher, True)

    total = 0.0
    n = 0
    for raw_batch in loader:
        data = batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            video=data["video"],  # type: ignore[arg-type]
            ppg=data["ppg"],
            gsr=data["gsr"],
            lengths=data["lengths"],
            temperature=cfg.temperature,
            teacher_detach=True,
        )
        loss, _ = student_ablation_loss(outputs, data["label"], spec, cfg)  # type: ignore[arg-type]
        loss.backward()
        if cfg.grad_clip_norm is not None:
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip_norm)
        optimizer.step()
        b = int(data["label"].shape[0])  # type: ignore[union-attr]
        total += float(loss.detach().cpu()) * b
        n += b
    return total / max(n, 1)


@torch.no_grad()
def validation_loss(
    model: AblationModel,
    loader: Iterable[object],
    spec: VariantSpec,
    cfg: AblationTrainConfig,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for raw_batch in loader:
        data = batch_to_device(raw_batch, device)
        outputs = model(
            video=data["video"],  # type: ignore[arg-type]
            ppg=data["ppg"],
            gsr=data["gsr"],
            lengths=data["lengths"],
            temperature=cfg.temperature,
            teacher_detach=True,
        )
        loss, _ = student_ablation_loss(outputs, data["label"], spec, cfg)  # type: ignore[arg-type]
        b = int(data["label"].shape[0])  # type: ignore[union-attr]
        total += float(loss.cpu()) * b
        n += b
    return total / max(n, 1)


@torch.no_grad()
def evaluate_video_only(
    model: AblationModel,
    loader: Iterable[object],
    num_classes: int,
    device: torch.device,
) -> Metrics:
    """Evaluate exactly the deployment path: video -> student -> emotion class."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for raw_batch in loader:
        data = batch_to_device(raw_batch, device)
        student_out = model.student(data["video"], lengths=data["lengths"])  # type: ignore[arg-type]
        pred = torch.argmax(student_out["logits"], dim=-1)
        y_true.extend(data["label"].detach().cpu().tolist())  # type: ignore[union-attr]
        y_pred.extend(pred.detach().cpu().tolist())
    return metrics_from_predictions(y_true, y_pred, num_classes)


def train_variant(
    spec: VariantSpec,
    train_loader: Iterable[object],
    val_loader: Iterable[object],
    test_loader: Iterable[object],
    cfg: AblationTrainConfig,
    base_student_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[AblationResult, AblationModel]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    model_cfg = ModelConfig(
        num_classes=cfg.num_classes,
        pretrained_resnet=cfg.pretrained_resnet,
        strict_mamba=cfg.strict_mamba,
        dropout=0.4,
        cag_threshold=cfg.conflict_threshold,
        temperature=cfg.temperature,
        alpha_ce=cfg.alpha_ce,
        beta_kd=cfg.beta_kd,
    )
    model = AblationModel(model_cfg, spec).to(device)
    if base_student_state is not None:
        model.student.load_state_dict(base_student_state, strict=True)

    output_dir = Path(cfg.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Physiological teacher is trained first, then frozen for controlled KD.
    if model.teacher is not None:
        train_teacher(
            model,
            train_loader,
            val_loader,
            cfg,
            device,
            ckpt_dir / f"{spec.name}_teacher.pt",
        )

    # Student + optional CAG/fused classifier parameters only.
    if model.teacher is not None:
        freeze_module(model.teacher, True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.max_epochs), eta_min=cfg.min_learning_rate
    )

    best_val = math.inf
    best_epoch = 0
    bad_epochs = 0
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, cfg.max_epochs + 1):
        _train_one_student_epoch(model, train_loader, optimizer, spec, cfg, device)
        val_loss = validation_loss(model, val_loader, spec, cfg, device)
        scheduler.step()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = ckpt_dir / f"{spec.name}_best.pt"
    torch.save(
        {
            "variant": spec.name,
            "model_state_dict": model.state_dict(),
            "train_config": asdict(cfg),
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        checkpoint_path,
    )

    metrics = evaluate_video_only(model, test_loader, cfg.num_classes, device)
    result = AblationResult(
        variant=VARIANT_DISPLAY_NAMES[spec.name],
        ppg=spec.use_ppg,
        gsr=spec.use_gsr,
        video=True,
        kd=spec.use_kd,
        cag=spec.use_cag,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        n_test=metrics.n,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        checkpoint=str(checkpoint_path),
    )
    return result, model


# ---------------------------------------------------------------------------
# CAG behavior analysis
# ---------------------------------------------------------------------------


@torch.no_grad()
def analyze_cag_behavior(
    model: AblationModel,
    loader: Iterable[object],
    device: torch.device,
    threshold: float,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, Union[str, int, float]]]:
    if not model.spec.use_cag:
        raise ValueError("CAG behavior analysis requires the full pkd_cag variant.")

    buckets: Dict[int, Dict[str, List[float]]] = {}
    model.eval()
    for raw_batch in loader:
        data = batch_to_device(raw_batch, device)
        outputs = model(
            video=data["video"],  # type: ignore[arg-type]
            ppg=data["ppg"],
            gsr=data["gsr"],
            lengths=data["lengths"],
            teacher_detach=True,
        )
        y = data["label"].detach().cpu().numpy()  # type: ignore[union-attr]
        delta = outputs["conflict_delta"].detach().cpu().numpy()
        omega = outputs["omega_physio"].detach().cpu().numpy()
        for cls, d, w in zip(y, delta, omega):
            cls = int(cls)
            bucket = buckets.setdefault(cls, {"delta": [], "omega": []})
            bucket["delta"].append(float(d))
            bucket["omega"].append(float(w))

    rows: List[Dict[str, Union[str, int, float]]] = []
    for cls in sorted(buckets):
        deltas = np.asarray(buckets[cls]["delta"], dtype=np.float64)
        omegas = np.asarray(buckets[cls]["omega"], dtype=np.float64)
        name = class_names[cls] if class_names is not None and cls < len(class_names) else str(cls)
        rows.append(
            {
                "class_id": cls,
                "emotion": name,
                "conflict_score": float(deltas.mean()) if len(deltas) else float("nan"),
                "high_conflict_windows_percent": float(100.0 * (deltas > threshold).mean()) if len(deltas) else float("nan"),
                "cag_physiological_weight": float(omegas.mean()) if len(omegas) else float("nan"),
                "n": int(len(deltas)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _write_dict_rows(path: Union[str, Path], rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_results(results: Sequence[AblationResult], output_dir: Union[str, Path]) -> None:
    output_dir = Path(output_dir)
    rows = [asdict(r) for r in results]
    _write_dict_rows(output_dir / "ablation_results.csv", rows)

    # Map display names back to canonical names for reference comparison.
    inverse = {display: name for name, display in VARIANT_DISPLAY_NAMES.items()}
    comparison: List[Dict[str, object]] = []
    for result in results:
        canonical = inverse[result.variant]
        ref = MANUSCRIPT_REFERENCE_RESULTS[canonical]
        comparison.append(
            {
                "configuration": result.variant,
                "experimental_accuracy": round(result.accuracy, 4),
                "manuscript_accuracy_reference": ref["accuracy"],
                "accuracy_delta_exp_minus_ref": round(result.accuracy - ref["accuracy"], 4),
                "experimental_f1": round(result.f1, 4),
                "manuscript_f1_reference": ref["f1"],
                "f1_delta_exp_minus_ref": round(result.f1 - ref["f1"], 4),
            }
        )
    _write_dict_rows(output_dir / "manuscript_reference_comparison.csv", comparison)


def print_results_table(results: Sequence[AblationResult]) -> None:
    header = (
        f"{'Configuration':44s} {'PPG':>3s} {'GSR':>3s} {'KD':>3s} {'CAG':>3s} "
        f"{'Acc%':>8s} {'Prec%':>8s} {'Rec%':>8s} {'F1%':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.variant:44s} {('Y' if r.ppg else 'N'):>3s} {('Y' if r.gsr else 'N'):>3s} "
            f"{('Y' if r.kd else 'N'):>3s} {('Y' if r.cag else 'N'):>3s} "
            f"{r.accuracy:8.2f} {r.precision:8.2f} {r.recall:8.2f} {r.f1:8.2f}"
        )


def run_all_ablations(
    train_loader: Iterable[object],
    val_loader: Iterable[object],
    test_loader: Iterable[object],
    cfg: Optional[AblationTrainConfig] = None,
    variants: Sequence[VariantName] = VARIANT_ORDER,
    class_names: Optional[Sequence[str]] = None,
) -> List[AblationResult]:
    """
    Run the controlled five-configuration ablation study.

    All variants start from the same initial video-student weights to improve
    comparability. Physiological teachers are trained per configuration.
    """
    cfg = cfg or AblationTrainConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create one initial student state and reuse it in every configuration.
    set_seed(cfg.seed)
    base_model_cfg = ModelConfig(
        num_classes=cfg.num_classes,
        pretrained_resnet=cfg.pretrained_resnet,
        strict_mamba=cfg.strict_mamba,
        dropout=0.4,
        cag_threshold=cfg.conflict_threshold,
        temperature=cfg.temperature,
        alpha_ce=cfg.alpha_ce,
        beta_kd=cfg.beta_kd,
    )
    base_student = VideoStudent(base_model_cfg)
    base_student_state = copy.deepcopy(base_student.state_dict())
    del base_student

    results: List[AblationResult] = []
    full_model: Optional[AblationModel] = None
    for variant_name in variants:
        spec = VARIANTS[variant_name]
        print(f"\n=== Running ablation: {VARIANT_DISPLAY_NAMES[variant_name]} ===")
        result, model = train_variant(
            spec,
            train_loader,
            val_loader,
            test_loader,
            cfg,
            base_student_state=base_student_state,
        )
        results.append(result)
        if variant_name == "pkd_cag":
            full_model = model

    write_ablation_results(results, output_dir)
    print("\nAblation results (video-only deployment metrics):")
    print_results_table(results)

    if full_model is not None:
        device = resolve_device(cfg.device)
        full_model = full_model.to(device)
        behavior_rows = analyze_cag_behavior(
            full_model,
            test_loader,
            device=device,
            threshold=cfg.conflict_threshold,
            class_names=class_names,
        )
        _write_dict_rows(output_dir / "cag_behavior.csv", behavior_rows)

    return results


# ---------------------------------------------------------------------------
# Optional LOSO helper
# ---------------------------------------------------------------------------


def run_loso_ablations(
    subject_ids: Sequence[object],
    loader_factory,
    cfg: Optional[AblationTrainConfig] = None,
    variants: Sequence[VariantName] = VARIANT_ORDER,
    class_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """
    Run the same ablations with Leave-One-Subject-Out (LOSO) cross-validation.

    ``loader_factory(held_out_subject)`` must return
        (train_loader, val_loader, test_loader)
    where the held-out subject appears ONLY in test_loader.

    Results are averaged across subjects and written to
    ``loso_ablation_summary.csv``. This helper is separate from Table IV so you
    can evaluate either the standard ablation split or the LOSO
    generalization analysis without mixing protocols.
    """
    cfg = cfg or AblationTrainConfig()
    root = Path(cfg.output_dir)
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, subject in enumerate(subject_ids):
        train_loader, val_loader, test_loader = loader_factory(subject)
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.output_dir = str(root / f"loso_subject_{subject}")
        # Vary seed slightly by fold while keeping it shared across variants.
        fold_cfg.seed = cfg.seed + fold_idx
        fold_results = run_all_ablations(
            train_loader,
            val_loader,
            test_loader,
            fold_cfg,
            variants=variants,
            class_names=class_names,
        )
        for result in fold_results:
            row = asdict(result)
            row["held_out_subject"] = str(subject)
            fold_rows.append(row)

    _write_dict_rows(root / "loso_ablation_folds.csv", fold_rows)

    summary: List[Dict[str, object]] = []
    for display_name in [VARIANT_DISPLAY_NAMES[v] for v in variants]:
        rows = [r for r in fold_rows if r["variant"] == display_name]
        if not rows:
            continue
        summary.append(
            {
                "configuration": display_name,
                "folds": len(rows),
                "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in rows])),
                "accuracy_std": float(np.std([float(r["accuracy"]) for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                "precision_mean": float(np.mean([float(r["precision"]) for r in rows])),
                "recall_mean": float(np.mean([float(r["recall"]) for r in rows])),
                "f1_mean": float(np.mean([float(r["f1"]) for r in rows])),
                "f1_std": float(np.std([float(r["f1"]) for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
            }
        )
    _write_dict_rows(root / "loso_ablation_summary.csv", summary)
    return summary


if __name__ == "__main__":
    print(
        "This file is an ablation-study module. Import run_all_ablations() and "
        "pass the train/validation/test DataLoaders. See the module docstring "
        "for the expected batch format."
    )
