# PKD-CAG

Research code accompanying **Beyond Facial Expressions: Cross-Modal Physiological Knowledge Distillation with Conflict-Aware Autonomic Gating for Video-Based Emotion Recognition**.

Muhammad Saleem and Deok-Hwan Kim — *IEEE Sensors Journal*, 2026. [Paper (DOI)](https://doi.org/10.1109/JSEN.2026.3750332)

## Overview

PKD-CAG transfers physiological information from a PPG/GSR teacher (1D-CNN and Mamba) to a facial-video student (ResNet-18 and Attention-LSTM). The conflict-aware autonomic gate modulates physiological supervision. At inference, only the video student is required.

## Repository structure

- `PKD_CAG_model.py`: model components, losses and synthetic-input smoke test.
- `PKD_CAG_ablation_study.py`: five ablation configurations, training and evaluation routines, LOSO helper and reference metrics.
- `data_acquisition/multimodal_data_collection.m`: basic biosignal and webcam acquisition.
- `data_acquisition/continuous_data_collection.m`: continuous recording, stimulus playback and event logging.
- `requirements.txt`: core Python dependencies (not a locked experimental environment).

## Installation

Use a Python environment with a mutually compatible PyTorch and torchvision installation appropriate for your CUDA hardware. See [PyTorch installation](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

For the Mamba architecture, install compatible versions of `mamba-ssm` and `causal-conv1d` following [Mamba installation instructions](https://github.com/state-spaces/mamba). Use `strict_mamba=True` for experiments using the Mamba architecture. The BiGRU fallback is not equivalent to Mamba.

## Model smoke test

```bash
python PKD_CAG_model.py
```

This uses synthetic tensors and KL distillation; it checks model execution, not paper accuracy or valence–arousal training.

## Ablation experiments

Prepare synchronized, preprocessed PyTorch DataLoaders before running the following example:

```python
from PKD_CAG_ablation_study import AblationTrainConfig, run_all_ablations

cfg = AblationTrainConfig(output_dir="ablation_results", strict_mamba=True)
results = run_all_ablations(train_loader, val_loader, test_loader, cfg)
```

Each batch should provide `video` [B,T,3,H,W], `ppg` [B,1,T_ppg], `gsr` [B,1,T_gsr], `label` [B], and `va` [B,2] for valence–arousal supervision. `lengths` [B] is optional. The main model also supports KL distillation, whereas the ablation module uses VA MSE by default. The ablation module pretrains the teacher; verify that the experimental schedule matches the paper before claiming exact reproduction.

## MATLAB data acquisition

The acquisition scripts record physiological streams through LabRecorder and facial video through a webcam. The continuous session script also presents stimulus clips and writes event timestamps.

**Requirements:** MATLAB (R2025b in the paper), MATLAB Support Package for USB Webcams, Image Processing Toolbox for `imshow`, EmotiBit streams available through LSL, and LabRecorder with its remote control server enabled (default `127.0.0.1:22345`).

From the repository root in MATLAB:

```matlab
run('data_acquisition/multimodal_data_collection.m')
```

For continuous stimulus presentation and session recording:

```matlab
run('data_acquisition/continuous_data_collection.m')
```

Set the participant/session identifier, output directory and recording settings before execution. Place authorized AVI/MP4 stimuli in `stimuli/` or configure `stimDir`. Configure the XDF output location separately in LabRecorder. The basic script records a configurable short session; the continuous script records webcam MP4, XDF via LabRecorder, and an event CSV with baseline, clip, rest and session timestamps. The continuous script uses local elapsed-time timestamps and approximate playback pacing, not hardware-verified synchronization or XDF event markers. Verify timing, actual frame rate, and signal alignment before experimental use. The scripts retain the original baseline/rest settings; review these against the study protocol.

## Results reported in the paper

| Configuration | Accuracy (%) | Macro F1 (%) |
| --- | ---: | ---: |
| Video student only | 85.79 | 84.62 |
| PPG-KD | 85.91 | 85.23 |
| GSR-KD | 87.89 | 85.93 |
| PPG+GSR KD without CAG | 90.02 | 89.83 |
| PKD-CAG | 92.29 | 92.01 |

These are reference values from Table IV, not results independently reproduced by this release. The separate subject-independent LOSO evaluation reports 78.43% mean accuracy and 78.01% macro F1.

## Data and reproducibility

Raw participant recordings, benchmark datasets, stimulus videos, pretrained checkpoints, dataset-specific preprocessing/loaders and exact split manifests are not included. Obtain data only through the appropriate permissions and access procedures. This repository provides model and experimental routines, not a complete end-to-end reproduction package.

## Citation

```bibtex
@article{saleem2026pkdcag,
  author = {Saleem, Muhammad and Kim, Deok-Hwan},
  title = {Beyond Facial Expressions: Cross-Modal Physiological Knowledge Distillation with Conflict-Aware Autonomic Gating for Video-Based Emotion Recognition},
  journal = {IEEE Sensors Journal},
  year = {2026},
  doi = {10.1109/JSEN.2026.3750332}
}
```

## License

No software license is specified. Contact the rights holders for permission to reuse or redistribute the code.
