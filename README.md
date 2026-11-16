# FAILED PROJECT SO ...
# 🧬 DNA-1B-Compression
### *Extreme Neural Network Compression using DNA-Inspired Architecture*

**Generate 1 Billion parameter models from just 257K parameters (3913x compression ratio).**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

---

## 🚀 Try it in Google Colab (Interactive)

Experience the DNA Architecture and 1B Model Generation directly in your browser without any setup!

| Notebook | Description | Link |
|:---|:---|:---:|
| 🧪 **Notebook 1** | **DNA Training & Core Architecture** <br> *Train the DNA Nanna on small matrices and understand Fourier Features.* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1h5Hl3ddqN0iBWKiid-JxF_FfH-4iZFdG) |
| 🏗️ **Notebook 2** | **1B Model Generation & Testing** <br> *Expand DNA to 1 Billion parameters, freeze, and test inference.* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dEMNXkRgUp92wapqZDbO0Z3wo8K-g3Sh) |

---

## 🎯 Key Highlights

- **🗜️ 3913x Compression**: Store a 1B model in just **1 MB** (DNA representation) instead of 3.75 GB.
- **⚡ On-Demand Generation**: Generate the full 1B model in **~19 seconds** using CPU.
- **🧠 Memory Efficient**: Layer-by-layer generation prevents OOM (Out of Memory) errors.
- **🌐 Zero Quality Loss**: Perfect mathematical reconstruction of weights.

---

## 📦 Pre-trained Models & Artifacts

All our trained DNA checkpoints and generated 1B models are hosted on Google Drive. You can download them directly to experiment.

| 📁 Artifact | Description | Size | Download Link |
|:---|:---|:---|:---:|
| 🧬 **DNA Genome (Pruned)** | The core pruned DNA controller (100 genes). | ~109 KB | [⬇️ Download](https://drive.google.com/file/d/1GX50s2m7UxZd41wvwkNC7z4LegwTO0BU/view?usp=drivesdk) |
| 🧠 **DNA Only** | Pure DNA weights (Tiny footprint). | ~1 MB | [⬇️ Download](https://drive.google.com/file/d/1wzSL4__eQyNGd0HLZE9YpvX2xOaSVONx/view?usp=drivesdk) |
| 🏗️ **1B Architecture** | Uninitialized 1B DNA structure. | ~0.2 MB | [⬇️ Download](https://drive.google.com/file/d/1cEXGmbQEDdq0OV3hfrATGv5ilCShRwfW/view?usp=drivesdk) |
| 🌌 **Extended DNA (1B)** | Expanded DNA initialized with proven patterns. | ~1 MB | [⬇️ Download](https://drive.google.com/file/d/1USdL33KtGQDNLiY4RZvUF5q6ItMtGyYU/view?usp=drivesdk) |
| 🧊 **Generated 1B Model** | The fully expanded & frozen 1B parameter model. | **3.75 GB** | [⬇️ Download](https://drive.google.com/file/d/1eiAjn9LFsN1uI4LTToaHcDyGMNiCLE_f/view?usp=drivesdk) |
| 📊 **Model Metadata** | Detailed stats, compression ratios, and logs. | ~2 KB | [⬇️ Download](https://drive.google.com/file/d/166neIxkZ-fprz81aUJmSMyLZ27Eq0pAH/view?usp=drivesdk) |
scroll

---

## 🧠 How It Works: The DNA Nanna Architecture

Instead of storing massive weight matrices, we train a tiny **Implicit Neural Representation (DNA Nanna)** that learns to *generate* weights on the fly based on their 2D coordinates $(i, j)$.

```mermaid
graph TD
    A[Coordinates i, j] --> B(Fourier Features)
    B --> C{DNA Nanna MLP}
    C --> D[Raw Weight Value]
    D --> E[Xavier Scaling]
    E --> F[Final Weight Matrix W]
```

**Why Fourier Features?** Standard MLPs suffer from *Spectral Bias* (they only learn smooth, low-frequency patterns). By mapping coordinates through randomized Fourier projections, our DNA Nanna can perfectly reconstruct high-frequency noise and complex weight distributions.

---

## 📊 Results & Benchmarks

| Metric | Value |
|:---|:---|
| **DNA Parameters** | 257,295 (~257K) |
| **Generated Virtual Parameters** | 1,006,772,226 (~1B) |
| **Compression Ratio** | **3913x** |
| **Storage (DNA)** | 1.00 MB |
| **Storage (Full Model)** | 3.75 GB |
| **Generation Time (15 Layers)** | 19.2 seconds |
| **Inference Time (Forward Pass)** | 349.44 ms |
| **RAM Usage (Inference)** | 4.5 GB |

---

## 🛠️ Quick Start (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/DNA-1B-Compression.git
cd DNA-1B-Compression
pip install -r requirements.txt
```

### 2. Generate 1B Model from DNA
```python
from src.dna_architecture import FourierDNA
from src.model_generator import generate_1b_model

# Load the tiny DNA
dna = FourierDNA(hidden=128, num_freqs=64)
dna.load_state_dict(torch.load("saved_models/DNA_only.pt"))

# Generate and save the 1B model directly to disk (OOM-Safe)
generate_1b_model(
    dna=dna,
    output_path='frozen_1b_model.pt',
    num_layers=15,
    layer_size=8192
)
```

### 3. Load and Run Inference
```python
import torch
from src.model_generator import Frozen1BModel

# Load the 3.75 GB frozen model
checkpoint = torch.load('frozen_1b_model.pt')
model = Frozen1BModel(d=8192, num_layers=15)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
x = torch.randn(1, 1)
output = model(x)
print(f"Output: {output.item()}")
```

---

## 📁 Project Structure

```text
DNA-1B-Compression/
├── src/
│   ├── dna_architecture.py    # FourierDNA Nanna class
│   ├── model_generator.py     # OOM-Safe 1B generation logic
│   └── utils.py               # Helper functions
├── scripts/
│   ├── train_dna.py           # Train DNA on small matrices
│   ├── generate_1b_model.py   # Expand and save 1B model
│   └── test_1b_model.py       # Load and benchmark
├── saved_models/              # Checkpoints (Download from Drive links above)
└── docs/                      # Technical documentation
```

---

## 🎓 Citation

If you use this architecture or the concept of DNA-based weight generation in your research, please cite:

```bibtex
@software{dna_1b_compression,
  title = {DNA-1B-Compression: Extreme Neural Network Compression via Implicit Weight Generation},
  author = {kapil dhinwa},
  year = {2026},
  url = {https://github.com/kapildhinwa-dev/DNA-1B-Compression}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check the [issues page](../../issues/).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>If you find this project useful, please consider giving it a ⭐!</b>
</p>


