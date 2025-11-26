import sys
import os
import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
except ImportError:
    raise SystemExit("Missing dependency: install gensim with `pip install gensim` before running this script.")

sys.path.append('Components')
from Tokenizer import tokenize

print("=" * 80)
print("Dense Vectors with Word2Vec (Component 1)")
print("=" * 80)

# Load movie datasets
decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
dataframes = []
for decade in decades:
    path = os.path.join('Data', f'{decade}-movies.csv')
    df = pd.read_csv(path)
    df['decade'] = decade
    dataframes.append(df)
    print(f"✓ Loaded {len(df)} movies from {decade}")

all_movies = pd.concat(dataframes, ignore_index=True)
print(f"\nTotal movies: {len(all_movies)}")

# Tokenize title + plot for each movie using the existing tokenizer (IIR-style)
print("\nTokenizing documents...")
all_movies['tokens'] = all_movies.apply(
    lambda row: tokenize(str(row['title']) + ' ' + str(row['plot']),
                         remove_stopwords=True,
                         apply_stemming=True),
    axis=1
)
sentences = all_movies['tokens'].tolist()
total_tokens = sum(len(tokens) for tokens in sentences)
print(f"✓ Prepared {len(sentences)} tokenized documents")
print(f"✓ Total tokens: {total_tokens:,}")

# Train Word2Vec to learn dense vectors for each token
print("\nTraining Word2Vec model...")
vector_size = 100
window = 5
min_count = 2
workers = os.cpu_count() or 4
model = Word2Vec(
    sentences=sentences,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    sg=1,          # use skip-gram (better for rare words)
    epochs=20
)
print(f"✓ Vocabulary size: {len(model.wv.index_to_key)} tokens")
print(f"✓ Embedding dimension: {vector_size}")

# Build a dense vector per document by averaging its token vectors
print("\nAveraging token vectors per document...")
doc_vectors = []
zeros = np.zeros(vector_size, dtype=np.float32)
for tokens in sentences:
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if vectors:
        doc_vectors.append(np.mean(vectors, axis=0))
    else:
        doc_vectors.append(zeros)

doc_matrix = np.vstack(doc_vectors)
print(f"✓ Document matrix shape: {doc_matrix.shape}")

# Quick peek at the first document vector
print("\nSample document:")
print(f"Title: {all_movies.iloc[0]['title']}")
print(f"Decade: {all_movies.iloc[0]['decade']}")
print(f"Token count: {len(all_movies.iloc[0]['tokens'])}")
print(f"First 10 dims of its vector: {doc_matrix[0][:10]}")

print("\nDone. The variables `model` (word embeddings) and `doc_matrix` (dense document vectors) are ready for the next stage.")
