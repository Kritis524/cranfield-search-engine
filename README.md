# Cranfield Search Engine

**Live demo:** https://cranfield-search-kritis.streamlit.app/

An information retrieval system over the Cranfield collection (1,400 aeronautical abstracts, 225 queries with human relevance judgments). It implements and compares four ranking methods, and includes an interactive search interface for trying them side by side.

## Ranking methods

| Method | Description |
|---|---|
| **TF-IDF** | Classical vector space model with cosine similarity |
| **BM25** | Probabilistic sparse retrieval with term-frequency saturation and document-length normalization |
| **Sentence Transformer** | Dense semantic retrieval using `all-MiniLM-L6-v2` embeddings and cosine similarity |
| **Hybrid (RRF)** | Reciprocal Rank Fusion of the TF-IDF, BM25 and dense rankings |

Reciprocal Rank Fusion combines rankings without needing score normalization:

```
RRF(d) = Σ  1 / (k + rank_m(d))      k = 60, summed over the ranking methods m
```

## Results

Evaluated on all 225 queries. Precision and nDCG are measured at cutoff 10, Recall at cutoff 100.

| Method | Precision@10 | Recall@100 | MAP | nDCG@10 |
|---|---|---|---|---|
| TF-IDF | 0.2293 | 0.7493 | 0.2960 | 0.3700 |
| BM25 | 0.2324 | 0.7467 | 0.3056 | 0.3781 |
| Sentence Transformer | 0.2440 | 0.7756 | 0.3231 | 0.3955 |
| **Hybrid (RRF)** | **0.2644** | **0.7796** | **0.3393** | **0.4223** |

- BM25 slightly outperforms TF-IDF, thanks to term saturation and length normalization.
- The Sentence Transformer improves recall by finding relevant documents that share few exact words with the query.
- The hybrid ranking is best on all four metrics, as it combines keyword and semantic evidence.

Results are saved to `results/comparison_table.csv` and `results/pr_curve.png` when you run the evaluation.

## How it works

1. **Preprocessing:** lowercasing, tokenization, stop-word removal and Porter stemming.
2. **Indexing:** an inverted index (term → documents with frequencies), a TF-IDF matrix, BM25 statistics and document embeddings.
3. **Ranking:** each query is scored by the four methods.
4. **Evaluation:** Precision@10, Recall@100, MAP and nDCG@10, plus an 11-point interpolated Precision-Recall curve. Documents with a judgment grade above 0 count as relevant for Precision, Recall and MAP. The graded judgments are used for nDCG.

The query file and the relevance judgments number the queries differently. The judgments use 1 to 225 by position, so the code maps queries by position.

## Search interface

The Streamlit app has two tabs:

- **Search:** type a query or pick one of the 225 benchmark queries. All four rankings appear side by side, and for benchmark queries the documents judged relevant are marked with ✅.
- **Evaluation results:** the comparison table and the Precision-Recall curves.


## Project structure

| Path | Purpose |
|---|---|
| `ir_core.py` | Data loading, preprocessing, inverted index, rankers, RRF and metrics |
| `evaluate.py` | Runs all methods on the 225 queries and writes the table and graph |
| `app.py` | Streamlit search interface |
| `data/` | Cranfield documents, queries and relevance judgments |
| `results/` | Comparison table and Precision-Recall graph |