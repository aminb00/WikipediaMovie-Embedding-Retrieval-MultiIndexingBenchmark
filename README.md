Here is a **clean, professional, and improved English version** of your project README.
I preserved all technical details but improved clarity, structure, and academic tone.

---

# Information Retrieval – High-Dimensional Similarity Search

This repository contains the full assignment pipeline for building dense vector representations of movie plot summaries and benchmarking multiple Approximate Nearest Neighbor (ANN) algorithms (VQ, PQ, LSH, FAISS).
All intermediate vectors and experiment outputs are stored in `Data/results`, and the notebooks document every step needed to reproduce the benchmarks.

---

## Repository Structure

### **Component1.ipynb – Embedding Construction**

Creates Word2Vec embeddings from the movie CSV files and saves:

* `Data/processed/doc_vectors_w2v.npy`
* `Data/processed/doc_metadata.csv`

### **Component2a.ipynb – Vector Quantization (VQ)**

Benchmark of **K-Means–based Vector Quantization**:

* Varies `n_probes`
* Tests scaling with dataset size and dimensionality
* Saves results as `vq_*` CSV files and corresponding plots

### **Component2b.ipynb – Product Quantization (PQ)**

Benchmark of **PQ**:

* Varies the number of sub-vectors `m`
* Measures accuracy vs. search speed and memory efficiency
* Saves `pq_*` CSV files in `Data/results`

### **Component3.ipynb – Locality-Sensitive Hashing (LSH)**

Custom LSH implementation using:

* Random hyperplanes
* Banding (b bands)

Benchmarks recall, candidate ratio, and scaling behaviour.
Outputs `lsh_*` CSV files in `Data/results`.

### **Component3b.ipynb – LSH Query Example**

Single-query demonstration of LSH:

* Shows candidate sets, recall, thresholds
* Logs stored in `Data/results/lsh_benchmark_runs.csv`

### **Component4.ipynb – FAISS Benchmark**

Benchmarks industrial ANN methods using **FAISS**:

* `IndexIVFFlat` (VQ)
* `IndexIVFPQ` (VQ + PQ)
* `IndexLSH`
* Compared against exact nearest neighbours

Stores results as `faiss_*` CSV files.

### **Component5.ipynb – Final Comparison**

Aggregates results from Components 2–4 and produces final comparison plots:

* `comparison_accuracy_efficiency.png`
* `comparison_scaling_N.png`
* `comparison_scaling_dim.png`

---

## Source Modules (`Components/`)

* **Tokenizer.py** – Word tokenization with NLTK, stopword removal, Porter stemming
* **vector_quantization.py** – Reusable VQ/K-Means implementation
* **product_quantization.py** – Implementation of PQ
* **lsh.py** – Random-hyperplane LSH with banding
* **word2vec_component.py** – Standalone script equivalent to Component1


---

## Data Folder

`Data/` contains:

* Raw movie CSV files (grouped by decade)
* Preprocessed vectors in `processed/`
* All benchmark outputs (CSV + PNG) in `results/`

---

## Requirements

Python 3 with the following libraries:

* `numpy`, `pandas`, `scikit-learn`
* `gensim` (Word2Vec)
* `nltk`
* `matplotlib`, `seaborn`
* `faiss-cpu`
* Jupyter Notebook

Component1 automatically downloads required NLTK resources (`punkt`, `stopwords`) if missing.

### Example setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn gensim nltk matplotlib seaborn faiss-cpu
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## How to Run the Pipeline

1. **Run `Component1.ipynb`** (or `Components/word2vec_component.py`)
   → Generates normalized document vectors in `Data/processed/`.

2. **Run Component2a, Component2b, and Component3**
   → Produces VQ, PQ, and LSH benchmarks.

3. *(Optional)*

   * `Component3b` for a focused LSH example
   * `Component4` for FAISS benchmarking
   * `Component5` for the final aggregated comparison plots

All generated CSVs and plots remain in `Data/results`, so you can inspect or reuse results without rebuilding experiments.


