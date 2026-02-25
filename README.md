# Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2502.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2502.xxxxx)
[![Project Page](https://img.shields.io/badge/🌐-Project%20Page-blue)](https://your-project-page.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

<h3>
<a href="https://arxiv.org/abs/2502.xxxxx">📄 Paper</a> |
<a href="https://your-project-page.github.io">🌐 Project Page</a> |
<a href="#-citation">📖 Citation</a>
</h3>

**[Yining Hong¹](https://evelinehong.github.io/)** · **[Huang Huang](https://qingh097.github.io/)¹** · **[Manling Li²](https://limanling.github.io/)** · **[Li Fei-Fei¹](https://profiles.stanford.edu/fei-fei-li)** · **[Jiajun Wu¹](https://jiajunwu.com/)** · **[Yejin Choi¹](https://homes.cs.washington.edu/~yejin/)**

¹Stanford University · ²Northwestern University

---

<img src="figures/reflection_teaser.png" width="100%">

</div>


## 🏠 BEHAVIOR-1K: Long-Horizon Household Tasks


<details open>
<summary><b>📋 Setup Instructions</b></summary>

### Step 1: Check CUDA Version

```bash
# Check CUDA version
nvcc --version
nvidia-smi
```

### Step 2: Navigate to BEHAVIOR Directory

```bash
cd BEHAVIOR
```

### Step 3: Configure CUDA Version

**⚠️ IMPORTANT**: Edit `setup.sh` and update the CUDA version to match your system.

```bash
# Open setup.sh with your preferred editor
nano setup.sh
# or
vim setup.sh
```

Find and modify the CUDA version line:
```bash
# Example: Change from
CUDA_VERSION="11.8"
# To your version:
CUDA_VERSION="12.1"  # Match your system's CUDA version
```

### Step 4: Run Setup Script

```bash
./setup.sh --new-env --omnigibson --bddl --dataset --primitives
```

This will:
- Create a new conda environment
- Install OmniGibson
- Install BDDL (Behavior Domain Definition Language)
- Download BEHAVIOR-1K dataset (~50GB)
- Install primitive actions

**Note**: Our version includes modifications to:
- `omnigibson.utils.object_state_utils.sample_kinematics` for improved object placement
- `setup.sh` for CUDA compatibility

### Step 5: Install BEHAVIOR Model

```bash
cd ../BEHAVIOR-model
pip install -e .
```

### Step 6: Verify Installation

```bash
python -c "import omnigibson as og; print(og.__version__)"
python -c "import omnigibson as og; print(og.DATASET_PATH)"
```

</details>

---

## 🗄️ MuJoCo Cupboard Fitting

<details open>
<summary><b>📋 Setup Instructions</b></summary>

### Step 1: Navigate to Cupboard Directory

```bash
cd cupboard
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv mujoco-env

# Activate environment
source mujoco-env/bin/activate  # On Linux/Mac
# mujoco-env\Scripts\activate  # On Windows
```

### Step 3: Install Package

```bash
pip install -e .
```

### Step 4: Test Interactive Environment

```bash
cd roboworld/envs
python franka_cupboard_interactive.py
```

</details>

---


## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{hong2026reflective,
  title={Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs},
  author={Hong, Yining and Huang, Huang and Li, Manling and Fei-Fei, Li and Wu, Jiajun and Choi, Yejin},
  journal={arXiv preprint arXiv:2502.xxxxx},
  year={2026}
}
```

</div>
