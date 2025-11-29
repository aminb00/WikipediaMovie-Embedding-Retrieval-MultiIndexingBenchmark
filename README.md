# 🎬 High-Dimensional Similarity Search

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**Information Retrieval – Assignment 2**  
University of Antwerp | Academic Year 2025/2026

---

## 📋 Overview

This project implements and benchmarks **Approximate Nearest Neighbor (ANN)** indexing techniques for high-dimensional similarity search on movie plot embeddings.

### Methods Implemented:
- **Vector Quantization (VQ)** – K-means based inverted file index
- **Product Quantization (PQ)** – Subspace quantization for memory-efficient search
- **Locality-Sensitive Hashing (LSH)** – Random hyperplane hashing with banding
- **FAISS** – Production-grade library comparison (IVF, IVFPQ, LSH)

### Key Results:
| Method | Best Recall@10 | Query Time |
|--------|----------------|------------|
| VQ | 97.6% | 1.18 ms |
| LSH | 100% | 11.33 ms |
| PQ | 33.0% | 1.74 ms |
| FAISS-IVF | 98.8% | 83.5 μs |

---

## 👥 Team

| Name | Student ID |
|------|------------|
| Alperen Davran | s0250946 |
| Matteo Carlo Comi | s0259766 |
| Shakhzodbek Bakhtiyorov | s0242661 |
| Amin Borqal | s0259707 |

---

## 📁 Repository Structure

```
Information-Retrieval-Assignment-2/
├── Components/                    # Reusable Python modules
│   ├── Tokenizer.py              # Text preprocessing (NLTK)
│   ├── vector_quantization.py    # VQ/K-means implementation
│   ├── product_quantization.py   # PQ implementation
│   ├── lsh.py                    # LSH with random hyperplanes
│   ├── evaluation.py             # IR metrics (Manning et al.)
│   └── word2vec_component.py     # Standalone embedding script
│
├── Data/
│   ├── raw/                      # Original movie CSV files
│   ├── processed/                # Word2Vec embeddings
│   │   ├── doc_vectors_w2v.npy
│   │   └── doc_metadata.csv
│   └── results/                  # Benchmark outputs (CSV + PNG)
│
├── Documentation/
│   ├── main.tex                  # LaTeX report source
│   ├── main.pdf                  # Compiled report (21 pages)
│   └── images/                   # Figures for the report
│
├── Component1.ipynb              # Embedding construction
├── Component2a.ipynb             # VQ benchmark
├── Component2b.ipynb             # PQ benchmark
├── Component3.ipynb              # LSH benchmark
├── Component4.ipynb              # FAISS benchmark
├── Component5.ipynb              # Final comparison
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Notebooks

| Notebook | Description | Outputs |
|----------|-------------|---------|
| **Component1** | Creates Word2Vec embeddings (200-dim) from 17,830 movie plots | `doc_vectors_w2v.npy` |
| **Component2a** | VQ benchmark: varies `n_probe`, tests scaling | `vq_*.csv`, plots |
| **Component2b** | PQ benchmark: varies subspaces `m`, measures compression | `pq_*.csv`, plots |
| **Component3** | LSH benchmark: varies bands/rows, measures recall | `lsh_*.csv`, plots |
| **Component4** | FAISS comparison: IVF, IVFPQ, LSH indices | `faiss_*.csv`, plots |
| **Component5** | Aggregated comparison across all methods | `comparison_*.png`, `summary_*.png` |

---

## 📊 Evaluation Metrics

Following Manning et al. *"Introduction to Information Retrieval"* (Chapter 8):

- **Recall@k** – Fraction of true top-k neighbors retrieved
- **Precision@k** – Equal to Recall@k in ANN context
- **nDCG@k** – Normalized Discounted Cumulative Gain (ranking quality)
- **Candidate Ratio** – Fraction of corpus examined
- **Query Time** – Average time per query
- **Build Time** – Index construction time

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/aminb00/Information-Retrieval-Assignment-2.git
cd Information-Retrieval-Assignment-2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the Pipeline

```bash
# Step 1: Generate embeddings
jupyter notebook Component1.ipynb

# Step 2: Run individual benchmarks
jupyter notebook Component2a.ipynb  # VQ
jupyter notebook Component2b.ipynb  # PQ
jupyter notebook Component3.ipynb   # LSH
jupyter notebook Component4.ipynb   # FAISS

# Step 3: Generate comparison plots
jupyter notebook Component5.ipynb
```

### 3. Compile Report (optional)

```bash
cd Documentation
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
gensim>=4.3.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
faiss-cpu>=1.7.4
jupyter>=1.0.0
```

---

## 📈 Generated Plots

### Experiment 1: Recall vs Efficiency
- `exp1_recall_vs_candidate_ratio.png`
- `exp1_ndcg_vs_candidate_ratio.png`
- `exp1_recall_vs_query_time.png`

### Experiment 2: Scaling with N
- `exp2_build_time_vs_N.png`
- `exp2_query_time_vs_N.png`

### Experiment 3: Scaling with Dimensionality
- `exp3_build_time_vs_dim.png`
- `exp3_query_time_vs_dim.png`

### Summary
- `summary_best_recall.png`
- `summary_best_ndcg.png`
- `summary_build_time.png`
- `summary_query_time.png`
- `pr_ndcg_vs_recall.png`

---

## 📚 References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. [Online](https://nlp.stanford.edu/IR-book/)

2. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.

3. FAISS Library: [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

4. Wikipedia Movies Dataset: [Kaggle](https://www.kaggle.com/datasets/exactful/wikipedia-movies)

---

## 📄 License

This project is for educational purposes as part of the Information Retrieval course at the University of Antwerp.
