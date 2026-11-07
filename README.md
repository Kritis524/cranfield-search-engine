# Project 1 - Cranfield Search Engine (TF-IDF, BM25, Sentence Transformer, Hybrid RRF)

## Run (Python 3.9+)
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python evaluate.py        # prints table, writes results/comparison_table.csv + results/pr_curve.png
    streamlit run app.py      # live demo: 4 rankers side by side + evaluation tab

First run downloads all-MiniLM-L6-v2 (~90 MB) and caches doc embeddings in cache/ (~1 min on CPU).
No NLTK downloads needed.

## Files
- ir_core.py  : loading, preprocessing, inverted index, rankers, RRF, metrics
- evaluate.py : 225 queries -> P@10, Recall@100, MAP, nDCG@10 + PR curve
- app.py      : Streamlit search interface
- data/       : Cranfield docs (1,400), queries (225), qrels
