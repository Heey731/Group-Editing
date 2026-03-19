<div align="center">
<h2><font color="red">Group Editing</font>: Edit Multiple Images in One Go</h2>

[Yue Ma](placeholder_url), [Xinyu Wang](placeholder_url), [Qianli Ma](placeholder_url), [Qinghe Wang](placeholder_url), [Mingzhe Zheng](placeholder_url), [Xiangpeng Yang](placeholder_url), [Hao Li](placeholder_url), [Chongbo Zhao](placeholder_url), [Jixuan Ying](placeholder_url), [Hongyu Liu](placeholder_url), [Qifeng Chen](placeholder_url)

<strong>Research Project</strong>

<a href='https://arxiv.org/abs/your-paper-id'><img src='https://img.shields.io/badge/ArXiv-coming_soon-red'></a>
<a href='https://heey731.github.io/group-editing-page/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![GitHub](https://img.shields.io/github/stars/your-org/Group-Edit?style=social)](https://github.com/your-org/Group-Edit)

</div>

## 🎏 Abstract
<b>TL; DR: <font color="red">Group Editing</font> enables consistent editing across multiple images in one go by combining pseudo-video modeling with explicit geometry cues.</b>

<details><summary>CLICK for the full abstract</summary>

> Editing a set of related images with consistent subject identity, style, and structure is challenging due to viewpoint and pose variation. Group Editing reformulates multiple images as a pseudo-temporal sequence and leverages a video-generation prior to improve global consistency. To further enhance cross-view alignment, we integrate VGGT-based geometric correspondence and flow cues into the generation process. Our implementation uses a practical 5-stage pipeline, including mask extraction, input conversion, VGGT token extraction, flow estimation, and final Wan-VACE based generation. This repository provides the research code and engineering pipeline for reproducible group editing experiments.

</details>

## 📀 Demo Video

Demo videos and visual comparisons will be released soon.

## 📋 Changelog

- 2026.03 Initial public release of Group-Edit codebase

## 🚧 Todo

- [ ] Release more demo videos and cases
- [ ] Add one-command pipeline launcher
- [ ] Add config-driven path management (YAML/JSON)
- [ ] Add cleaner benchmark/evaluation scripts
- [ ] Release training details and model cards

## ✨ Features

- **Group-level consistent editing** across multiple input images
- **Pseudo-video reformulation** for improved temporal-like coherence
- **VGGT-based geometry guidance** for better correspondence alignment
- **Mask-aware subject editing** using GroundingDINO + SAM
- **Flow-guided generation** with multi-stage preprocessing
- **Wan-VACE + LoRA integration** for controllable generation

## 🛡 Setup Environment

```bash
# Create conda environment
conda create -n group-edit python=3.10
conda activate group-edit

# Install dependencies
cd Group-Edit
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU
- Recommended: 24GB+ VRAM (higher VRAM provides smoother inference)

## 📥 Model Download

This project needs several checkpoints from Hugging Face / ModelScope plus your project LoRA.

| Component | Model ID / Source | Local Target Directory | Used In |
|---|---|---|---|
| Grounding DINO | `IDEA-Research/grounding-dino-base` | `./models/IDEA-Research/grounding-dino-base` | `utils/process-origin2mask.py` |
| SAM | `facebook/sam-vit-huge` | `./models/facebook/sam-vit-huge` | `utils/process-origin2mask.py` |
| VGGT | `facebook/VGGT-1B` | `./models/facebook/models--facebook--VGGT-1B` | `vggt/infer-out-from-video-4frame.py` |
| Wan VACE 14B shards | `Wan-AI/Wan2.1-VACE-14B` | `./models/Wan-AI/Wan2.1-VACE-14B` | `infer-test.py` |
| Wan converted T5/VAE | `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors` | `infer-test.py` |
| Group-Edit LoRA | your trained checkpoint | `./models/epoch-9.safetensors` | `infer-test.py` |

### Download from Hugging Face (for GroundingDINO / SAM / VGGT)

```bash
python -m huggingface_hub download IDEA-Research/grounding-dino-base \
  --local-dir ./models/IDEA-Research/grounding-dino-base

python -m huggingface_hub download facebook/sam-vit-huge \
  --local-dir ./models/facebook/sam-vit-huge

python -m huggingface_hub download facebook/VGGT-1B \
  --local-dir ./models/facebook/models--facebook--VGGT-1B
```

### Download Wan checkpoints (example with ModelScope)

```bash
# Wan VACE 14B
modelscope download --model Wan-AI/Wan2.1-VACE-14B \
  --local_dir ./models/Wan-AI/Wan2.1-VACE-14B

# Wan converted safetensors (T5 + VAE)
modelscope download --model DiffSynth-Studio/Wan-Series-Converted-Safetensors \
  --local_dir ./models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
```

> Note: `infer-test.py` currently uses `ckpt_path` at line 45. Please set it to your local LoRA checkpoint path before running.

## ⚔️ Group Editing Inference

#### Quick Start (5-stage pipeline)

```bash
# 1) Extract object masks from origin videos
python utils/process-origin2mask.py

# 2) Convert mask/origin videos to pipeline input format
python utils/process-mask2input.py

# 3) Extract VGGT tokens
# Optional: export VGGT_MODEL_ROOT=./.model/facebook/models--facebook--VGGT-1B
python vggt/infer-out-from-video-4frame.py

# 4) Compute flow tensors from masks
python utils/2delta-batch-gpu-multi-frame.py

# 5) Run final generation
python infer-test.py
```

#### Input Data Format

- Origin video: `./test-data/Gemini-out/<id>-origin.mp4`
- Object description JSON: `./test-data/gemini-test.json`
- Generated intermediate folders:
  - `./test-data/Gemini-out-expand-5`
  - `./test-data/Gemini-out-expand-5-vggt`
  - `./test-data/Gemini-out-expand-5-map`

## 📁 Project Structure

<details><summary>Click for directory structure</summary>

```text
Group-Edit/
├── diffsynth/                         # Core diffusion framework
│   ├── models/                        # Model definitions (Wan DiT/VACE, encoders, etc.)
│   └── pipelines/                     # Pipeline implementations (wan_video_new.py)
├── utils/
│   ├── process-origin2mask.py         # Stage-1 mask extraction (GroundingDINO + SAM)
│   ├── process-mask2input.py          # Stage-2 input conversion
│   └── 2delta-batch-gpu-multi-frame.py# Stage-4 flow tensor extraction
├── vggt/
│   └── infer-out-from-video-4frame.py # Stage-3 VGGT token extraction
├── infer-test.py                      # Stage-5 final inference
├── models/                            # Local checkpoints (ignored by git)
├── test-data/                         # Local data and intermediate files (optional)
├── requirements.txt
└── README.md
```

</details>

## 🔧 Key Modifications

This repository is built on top of DiffSynth-Studio and includes project-specific edits for Group Editing:

### 1. `infer-test.py`

- Integrates LoRA loading for Wan-VACE pipeline
- Loads and injects VGGT tokens (`vggt_tensor`) and flow tensors (`flow_tensor`)
- Implements practical task loop for grouped editing generation

### 2. `vggt/infer-out-from-video-4frame.py`

- Adds masked-frame token extraction for video-style group inputs
- Supports Hugging Face cache-style model root resolution (`snapshots/<revision>`)

### 3. `utils/process-origin2mask.py` + `utils/2delta-batch-gpu-multi-frame.py`

- Stage-1 object mask extraction with GroundingDINO and SAM
- Stage-4 contour/TPS-based flow map generation for guidance

### 4. `diffsynth/models/stepvideo_text_encoder.py`

- Added import fallback for transformers API compatibility across versions:
  - `from transformers import PretrainedConfig, PreTrainedModel`
  - fallback to `configuration_utils` / `modeling_utils`

## 📍 Citation

If you use this code, please cite:

```bibtex
@article{groupediting2026,
  title={Group Editing: Edit Multiple Images in One Go},
  author={Ma, Yue and Wang, Xinyu and Ma, Qianli and Wang, Qinghe and Zheng, Mingzhe and Yang, Xiangpeng and Li, Hao and Zhao, Chongbo and Ying, Jixuan and Liu, Hongyu and Chen, Qifeng},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## 📜 License

This project is released under the Apache-2.0 License.
See [LICENSE](LICENSE) for details.

## 💗 Acknowledgements

This repository builds upon:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [VGGT](https://github.com/facebookresearch/vggt)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

Thanks to the original authors and communities for open-sourcing their work.

## 🧿 Maintenance

This repository is maintained for research and reproducibility.
If you find issues or have suggestions, please open an issue or discussion thread.
