# QFennecs: Quantum Attention for Molecular Classification

**Project for QPoland Global Quantum Hackathon 2025**

### Team QFennecs

- **Rihab Hoceini**
- **Raouf Ould Ali**
- **Amieur Lilya Fatima-Zohra**
- **Widad Hassina Belkadi**
- **Hamza Abderaouf Khentache**

***

## Overview

This project proposes an **integrated pipeline combining classical graph-based methods, attention mechanisms, and quantum computing** for **molecular classification**.
Designed for the **QPoland Global Quantum Hackathon 2025**, the solution enhances molecular property prediction using **Quantum Feature Maps** and **Attention-weighted Graph Embeddings**.

***

## Motivation

Molecular systems can be naturally modeled as graphs where **atoms are nodes** and **bonds are edges**. Understanding their structure is central to predicting chemical, biological, and physical properties.
Conventional graph methods struggle with fine‑grained chemical semantics. To address this, our work introduces:

- **Attention layers** for context‑aware atom and bond relevance.
- **Quantum encoders** for capturing entanglement‑aware molecular structures.
- **Hybrid feature maps** fusing classical learning and quantum computation.

***

## Methodology

### 1. Datasets

Benchmarks from **TUDataset** are employed:


| Dataset | Domain | Graphs | Avg. Nodes | Avg. Edges | Labels |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MUTAG | Chemistry | 188 | 17 | 19 | Mutagenic / Non‑Mutagenic |
| AIDS | Chemistry | 2000 | 15 | 16 | Active / Inactive |
| NCI1 | Chemistry | 4110 | 30 | 32 | Active / Inactive |
| PTC‑MR | Toxicology | 344 | 25 | 26 | Toxic / Non‑Toxic |
| PROTEINS | Biology | 1113 | 39 | 73 | Enzyme / Non‑Enzyme |

### 2. Representation

- Each molecule is an **undirected graph** built using **NetworkX**.
- **Nodes:** atom types (C, N, O, etc.)
- **Edges:** bond orders (1.0, 2.0, 3.0, or 1.5 for aromatic).


### 3. Feature Extraction

Three complementary modules operate sequentially:

1. **Chemical Laplacian Features:**
Spectral, topological, and structural embeddings based on Laplacian eigenvalues and hierarchical connectivity.
2. **Attention‑Based Features:**
Node‑ and edge‑level attention mechanisms emphasize functionally and chemically relevant substructures.
3. **Quantum Feature Maps (QURI Framework):**
Each molecule is transformed into a quantum state where:
    - Qubits represent atoms.
    - Controlled‑Z gates encode bonds.
Quantum observables (entanglement entropy, Hamiltonian energy, quantum walk overlap) yield entanglement‑aware embeddings.

### 4. Classification

A normalized feature matrix is trained with an **SVM (RBF kernel)** via **10‑fold cross‑validation**, tuned through grid search.

***

## Results

| Dataset | Chemical Laplacian (F1/Acc) | Attention‑Based (F1/Acc) | Quantum Attention‑Based (F1/Acc) |
| :-- | :-- | :-- | :-- |
| MUTAG | 0.80 / 0.82 | 0.83 / 0.85 | **0.90 / 0.86** |
| PTC‑MR | 0.52 / 0.59 | 0.51 / 0.58 | **0.53 / 0.60** |
| NCI1 | 0.70 / 0.73 | 0.73 / 0.74 | – |
| AIDS | 0.84 / 0.95 | 0.98 / 0.98 | **0.99 / 0.98** |
| PROTEINS | 0.63 / 0.66 | 0.60 / 0.67 | – |

The **Quantum Attention Model** demonstrates superior expressivity, interpretability, and domain generality.

***

## How to Run

Clone and execute all cells in either notebook:

```bash
git clone https://github.com/HoceiniRihab/QFennecs_QPoland-Global-Quantum-Hackathon.git
cd QFennecs_QPoland-Global-Quantum-Hackathon
```

Then open one of:

- [AttentionAndQuantumWithQuri.ipynb](https://github.com/HoceiniRihab/QFennecs_QPoland-Global-Quantum-Hackathon/blob/main/AttentionAndQuantumWithQuri.ipynb)
- [AttentionAndQuantum.ipynb](https://github.com/HoceiniRihab/QFennecs_QPoland-Global-Quantum-Hackathon/blob/main/AttentionAndQuantum.ipynb)

Execute cells sequentially.

***

## Presentations

🎥 **Video Presentation:** [Watch here](https://github.com/HoceiniRihab/QFennecs_QPoland-Global-Quantum-Hackathon/raw/refs/heads/main/Quantum%20Hack1.mp4)

📑 **Slides:** [Presentation.pdf](https://github.com/HoceiniRihab/QFennecs_QPoland-Global-Quantum-Hackathon/blob/main/Presentation.pdf)


