<div align="center">

<h2><font color="red">Group Editing</font>: Edit Multiple Images in One Go</h2>

<p>
  <a href="https://github.com/your-org/your-repo"><img src="https://img.shields.io/badge/Code-GitHub-black"></a>
  <a href="https://github.com/your-org/your-repo/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-brightgreen">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c">
  <img src="https://img.shields.io/badge/Status-Research%20Code-orange">
</p>

<p>
  <a href="https://arxiv.org/abs/your-paper-id"><img src="https://img.shields.io/badge/ArXiv-coming_soon-red"></a>
  <a href="https://your-project-page"><img src="https://img.shields.io/badge/Project-Page-green"></a>
</p>

</div>

---

> **TL;DR**: Group Editing is a framework for consistent multi-image editing that combines pseudo-video modeling, VGGT-based geometric correspondence, and Wan-VACE generation.

---

## 🎏 Abstract
Editing a group of related images consistently is challenging due to viewpoint changes, pose variation, and spatial misalignment.  

**Group Editing** reformulates image groups as pseudo-temporal sequences, enabling video generative priors to enforce implicit consistency. To strengthen explicit alignment, we integrate dense geometric correspondence extracted from VGGT features. We further introduce:

- **Ge-RoPE**: geometry-enhanced positional encoding for cross-view spatial alignment,
- **Identity-RoPE**: identity-aware positional encoding for stable subject consistency.

In practice, this repository provides a modular 5-stage pipeline (mask -> input formatting -> VGGT token extraction -> flow estimation -> final generation) that supports reproducible experiments and practical deployment.

---

## 🧠 Core Contributions

1. **Pseudo-video formulation for image groups**
- Treats multi-image editing as sequence modeling
- Leverages video diffusion priors for coherence

2. **Explicit geometric correspondence with VGGT**
- Extracts geometry-aware features across views
- Provides robust alignment cues for editing

3. **Dual positional alignment design**
- **Ge-RoPE** for geometric consistency
- **Identity-RoPE** for subject/identity preservation

4. **Engineering-friendly modular pipeline**
- Clear stage boundaries for debugging and ablations
- Compatible with Wan-VACE + LoRA workflow

---

## ✨ Features

- **Consistent group editing** across multiple images/frames
- **Hybrid alignment**: implicit temporal priors + explicit geometry priors
- **Mask-aware subject control** via GroundingDINO + SAM
- **Geometry-guided generation** via VGGT tokens + flow tensors
- **LoRA-based adaptation** for custom editing behavior
- **Script-level modularity** for fast experimentation

---

## 📋 Changelog

- 2026.03 Public open-source README release
- 2026.03 Integrated paper-level method description with practical usage guide
- 2026.03 Added path and troubleshooting sections for reproducibility

---

## 🚧 Todo

- [ ] Add demo GIFs / benchmark visualizations
- [ ] Add one-command launcher for the full pipeline
- [ ] Add YAML/JSON config system to replace hard-coded paths
- [ ] Add batch inference examples
- [ ] Add training/fine-tuning documentation

---

## 🛡 Environment Setup

### Option A (Recommended)

```bash
conda create -n group-edit python=3.10 -y
conda activate group-edit

cd Group-Editing
pip install -r requirements.txt
pip install -e .
```

### Option B (From `environment.yml`)

```bash
# If environment.yml contains a machine-specific prefix, remove/adjust it first.
conda env create -f environment.yml
conda activate group-edit
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU
- ffmpeg (recommended for stable video I/O)
- High VRAM recommended for Wan2.1-VACE-14B

---

## 📥 Model Preparation

Prepare the following checkpoints before running:

| Component | Model | Stage |
|---|---|---|
| Detection | `IDEA-Research/grounding-dino-base` | Mask extraction |
| Segmentation | `facebook/sam-vit-huge` | Mask extraction |
| Geometry token model | `facebook/VGGT-1B` | VGGT extraction |
| Video backbone | `Wan-AI/Wan2.1-VACE-14B` (7 shards) | Final inference |
| Text/VAE weights | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | Final inference |
| LoRA | `epoch-9.safetensors` (custom) | Final inference |

### Notes

- `VGGT-1B` may be stored in Hugging Face cache layout (`models--.../snapshots/<hash>`). The provided VGGT script supports this.
- `infer-test.py` supports env overrides:
  - `WAN_VACE_ROOT`
  - `WAN_CONVERTED_ROOT`
- If tokenizer files are missing, the pipeline may attempt to fetch `Wan-AI/Wan2.1-T2V-1.3B/google/*` automatically.

---

## 📁 Data Format

### Inputs

- Source videos: `./test-data/Gemini-out/{id}-origin.mp4`
- Subject metadata: `./test-data/gemini-test.json`

Example JSON:

```json
[
  {
    "description": {"item": "fox"},
    "image_filename": "351-origin.mp4"
  }
]
```

### Intermediate / Outputs

- Mask video: `./test-data/Gemini-out/{id}-mask.mp4`
- Formatted folder: `./test-data/Gemini-out-expand-5/`
- VGGT tokens: `./test-data/Gemini-out-expand-5-vggt/{id}-origin_aggregated_tokens.npy`
- Flow tensors: `./test-data/Gemini-out-expand-5-map/{id}-map.npy`
- Final outputs: `./test-out/*.mp4`

---

## ⚔️ Quick Start (End-to-End)

Run from repository root:

```bash
# 1) Extract masks from origin videos
python utils/process-origin2mask.py

# 2) Build formatted inputs
python utils/process-mask2input.py

# 3) Compute VGGT token tensors
python vggt/infer-out-from-video-4frame.py

# 4) Compute flow tensors
python utils/2delta-batch-gpu-multi-frame.py

# 5) Run final Wan-VACE + LoRA inference
python infer-test.py
```

---

## ⚙️ Key Path Configuration

Before running on a new machine, verify these scripts:

1. `utils/process-origin2mask.py`
- `JSON_FILE_PATH`
- `VIDEO_DIR`
- `detector_id`
- `segmenter_id`

2. `vggt/infer-out-from-video-4frame.py`
- `VGGT_MODEL_ROOT` (env override supported)
- `folder_path`

3. `utils/2delta-batch-gpu-multi-frame.py`
- `input_folder`

4. `infer-test.py`
- `video_base_path`
- `ckpt_path` (LoRA)
- `tasks`
- `WAN_VACE_ROOT` / `WAN_CONVERTED_ROOT`

---

## 🧪 Troubleshooting

### 1) `config.json/model.safetensors not found` (VGGT)
Cause: incorrect directory level (cache root vs snapshot) or incomplete download.  
Fix: point to valid VGGT model root/snapshot.

### 2) `ImportError` from `transformers.modeling_utils`
Cause: transformers version mismatch.  
Fix: use compatibility import patch in `diffsynth/models/stepvideo_text_encoder.py`.

### 3) `No such file or directory` for Wan checkpoints
Cause: path mismatch on new machine.  
Fix: update defaults or set env vars (`WAN_VACE_ROOT`, `WAN_CONVERTED_ROOT`).

### 4) CUDA initialization/runtime errors
Cause: GPU visibility/driver/container mismatch.  
Fix: verify `nvidia-smi`, CUDA runtime, and container launch settings.

---

## 📁 Project Structure

```text
Group-Editing/
├── diffsynth/                             # Core model and pipeline code
├── utils/
│   ├── process-origin2mask.py             # Stage 1: mask extraction
│   ├── process-mask2input.py              # Stage 2: input formatting
│   └── 2delta-batch-gpu-multi-frame.py    # Stage 4: flow computation
├── vggt/
│   └── infer-out-from-video-4frame.py     # Stage 3: VGGT extraction
├── infer-test.py                          # Stage 5: final inference
├── test-data/                             # Inputs and intermediates
├── models/                                # LoRA and local checkpoints
├── requirements.txt
└── setup.py
```

---

## 🔧 Reproducibility Checklist

- [ ] Model paths are valid on your machine
- [ ] Input naming follows `{id}-origin.mp4`
- [ ] `gemini-test.json` fields are correct
- [ ] Stage outputs exist before next stage
- [ ] GPU memory is sufficient for selected model variant

---

## 📍 Citation

If you use this repository, please cite Group Editing and the upstream projects.

```bibtex
@misc{group_editing_2026,
  title={Group Editing: Edit Multiple Images in One Go},
  author={Your Name and Contributors},
  year={2026},
  howpublished={\url{https://github.com/your-org/your-repo}}
}
```

---

## 📜 License

This project is released under the Apache-2.0 License.  
See [LICENSE](./LICENSE) for details.

---

## 💗 Acknowledgements

This project builds upon and/or references:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [VGGT](https://github.com/facebookresearch/vggt)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

---

## 🧿 Maintenance

This repository is released as research/engineering code.  
If you encounter issues, please open an issue with:

- full error logs,
- environment information,
- the exact stage command you executed.
