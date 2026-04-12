# 📰 Fake News Detection using Graph Machine Learning

A complete end-to-end project that applies **Graph Neural Networks (GCN, GAT, GraphSAGE)** to detect fake news by modeling relationships between articles, speakers, and subjects as a knowledge graph.

---

## 🧠 Project Overview

Traditional NLP approaches treat news articles in isolation. This project treats them as **nodes in a graph**, capturing the relational structure:

```
[Speaker] ──spoke──▶ [Article] ──about──▶ [Subject]
```

By propagating information across this graph, GNNs learn richer representations that capture credibility signals beyond just the text.

---

## 📂 Repository Structure

```
fake-news-graph-ml/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA, label distribution, word clouds
│   ├── 02_graph_construction.ipynb   # Build PyG graph, TF-IDF features
│   ├── 03_gnn_training.ipynb         # Train GCN, GAT, GraphSAGE
│   └── 04_visualization.ipynb        # t-SNE, ROC, attention heatmaps
├── data/                             # Created at runtime (gitignored)
├── models/                           # Saved model weights (gitignored)
└── results/                          # All output plots and metrics
    ├── tsne_embeddings.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── gat_attention.png
    ├── predicted_graph.png
    └── metrics.csv
```

---

## 🗂️ Dataset

**LIAR Dataset** — 12,836 labeled political statements from PolitiFact.

| Field | Description |
|-------|-------------|
| `statement` | The news claim text |
| `speaker` | Person who made the claim |
| `subject` | Topic of the claim |
| `label` | 6-class truthfulness label |

We binarize: `{pants-fire, false, barely-true}` → **Fake (1)**, `{half-true, mostly-true, true}` → **Real (0)**

Downloaded automatically via HuggingFace `datasets` library.

---

## 🕸️ Graph Construction

| Component | Details |
|-----------|---------|
| **Article nodes** | TF-IDF features (500-dim, bigrams) |
| **Speaker nodes** | One-hot encoded identity |
| **Subject nodes** | One-hot encoded topic |
| **Edges** | Speaker↔Article, Article↔Subject (bidirectional) |

---

## 🤖 Models

| Model | Architecture | Key idea |
|-------|-------------|----------|
| **GCN** | 2-layer Graph Conv + Linear | Averages neighbor features |
| **GAT** | 2-layer Graph Attention + Linear | Learns attention weights over neighbors |
| **GraphSAGE** | 2-layer SAGE + Linear | Samples and aggregates neighbors |

All models: hidden=128 → 64 → 2, dropout=0.5, Adam optimizer, 100 epochs.

---

## 📊 Results

| Model | Accuracy | F1 Score | AUC-ROC |
|-------|----------|----------|---------|
| GCN | ~0.63 | ~0.62 | ~0.67 |
| GAT | ~0.65 | ~0.64 | ~0.69 |
| GraphSAGE | ~0.64 | ~0.63 | ~0.68 |

*Results vary slightly across runs due to random initialization.*

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Upload notebooks to [colab.research.google.com](https://colab.research.google.com)
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run notebooks **in order**: `01 → 02 → 03 → 04`

### Option 2 — Local (VS Code)
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-graph-ml
cd fake-news-graph-ml
pip install -r requirements.txt
jupyter notebook
```

---

## 📦 Requirements

See `requirements.txt` for full list. Core dependencies:
- `torch` + `torch-geometric`
- `transformers` + `datasets`
- `scikit-learn`
- `networkx`
- `matplotlib` + `seaborn`
- `wordcloud`

---

## 📈 Visualizations

- **t-SNE plots** — shows how GNN embeddings cluster real vs fake news
- **Confusion matrices** — per-model breakdown of errors
- **ROC curves** — all 3 models compared
- **GAT attention heatmap** — which node relationships matter most
- **Knowledge graph** — final predictions overlaid on the graph structure

---

## 🏗️ Project by

Your Name | Course: ML on Graphs | Semester: 2025–26

---

## 📄 References

- Wang, W. Y. (2017). "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection. *ACL 2017*
- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*
- Veličković et al. (2018). Graph Attention Networks. *ICLR 2018*
- Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*
