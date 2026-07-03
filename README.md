# Hy-LIN: Hybrid Latent Interaction Network for Bug Triage

This repository provides the reference implementation accompanying the paper **"Hy-LIN: Reconstructing Latent Collaborations for Bug Triage with Dual-Path Graph Learning"**.

## 💡 Overview

Bug triage is a fundamental task in software maintenance. **Hy-LIN** addresses this by operating directly on native issue-tracking data without requiring manual feature engineering. 

The framework tackles the issue via two macro-level innovations:
1. **Learnable Graph Augmentation:** Dynamically injects sparse, semantic-guided latent interactions into the observed bug-developer bipartite graph.
2. **Dual-Path Propagation and Fusion:** Adaptively fuses topological representations from both explicit (bipartite) and implicit (developer-projected) structural views.

*(For detailed mathematical formulations, gating mechanisms, and BPR optimization strategies, please refer to the Methodology section of our main paper.)*

---

## 📌 Important Notes for Artifact Evaluation

### 1. Toy Dataset Declaration
To facilitate rapid reproducibility checks for reviewers and avoid distributing large-scale proprietary/community data, **the datasets included in the `data/` directory are heavily subsampled "Toy Datasets"**. They preserve the structural integrity required to verify the pipeline's execution but are not the full-scale datasets used in the paper.

For complete raw issue tracker data and standardized preprocessing protocols, please refer to the recent benchmark dataset formalized in:
> *Tao, Y., Dai, J., Ma, L., Ren, Z., & Wang, F. (2025). Structural contrastive learning based automatic bug triaging. Automated Software Engineering, 32(51). https://doi.org/10.1007/s10515-025-00517-5*

### 2. Performance Metrics Variation
You may observe varying numerical metrics when running this repository compared to the official paper. This is strictly expected because:
* The provided datasets are miniaturized subsets.
* The paper reports the most robust, conservative baselines averaged across multiple random seeds, rather than cherry-picked peaks.
* Post-submission codebase optimizations (e.g., streamlined sparse operations) are included here for better efficiency.

---

## 🚀 Quick Start

**Requirements:** Python 3.8+, PyTorch >= 1.12.

```bash
# 1. Clone the repository and install dependencies
pip install -r requirements.txt

# 2. Run the evaluation pipeline (Example: Chromium toy dataset)
python run.py --dataset gc
