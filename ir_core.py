"""Core IR pipeline for Project 1: Cranfield search engine.

Rankers: TF-IDF, BM25, Sentence Transformer (dense), Hybrid RRF.
No NLTK data downloads needed (regex tokenizer + sklearn stop-words + Porter stemmer).
"""
import os
import re
from collections import defaultdict

import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_stemmer = PorterStemmer()
_stem_cache = {}


# ---------------------------------------------------------------- 1. data
def load_docs():
    """Return (doc_ids, titles, texts). `<text>` already begins with the title."""
    raw = open(os.path.join(DATA_DIR, "cran.all.1400.xml"), encoding="utf-8", errors="ignore").read()
    ids, titles, texts = [], [], []
    for block in re.findall(r"<doc>(.*?)</doc>", raw, re.S):
        ids.append(int(re.search(r"<docno>\s*(\d+)\s*</docno>", block).group(1)))
        titles.append(" ".join(re.search(r"<title>(.*?)</title>", block, re.S).group(1).split()))
        texts.append(" ".join(re.search(r"<text>(.*?)</text>", block, re.S).group(1).split()))
    return ids, titles, texts


def load_queries():
    """Return {qrels_qid: query_text}.

    The qrels file numbers queries 1..225 by POSITION in the query file (the
    query file keeps the original Cranfield ids 1,2,4,...), so we map by order.
    """
    raw = open(os.path.join(DATA_DIR, "cran.qry.xml"), encoding="utf-8", errors="ignore").read()
    tops = re.findall(r"<top>.*?<title>(.*?)</title>", raw, re.S)
    return {i + 1: " ".join(t.split()) for i, t in enumerate(tops)}


def load_qrels():
    """Return {qid: {doc_id: grade}} keeping only grade > 0."""
    qrels = defaultdict(dict)
    for line in open(os.path.join(DATA_DIR, "cranqrel.trec.txt")):
        qid, _, did, rel = line.split()
        if int(rel) > 0:
            qrels[int(qid)][int(did)] = int(rel)
    return dict(qrels)


# ---------------------------------------------------------- 2. preprocessing
def preprocess(text):
    """Clean -> tokenize -> stop-word removal -> Porter stemming."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in ENGLISH_STOP_WORDS or len(t) < 2:
            continue
        if t not in _stem_cache:
            _stem_cache[t] = _stemmer.stem(t)
        out.append(_stem_cache[t])
    return out


# ---------------------------------------------------------------- 3. rankers
def rrf(rankings, k=60, depth=None):
    """Reciprocal Rank Fusion. rankings: list of ranked doc-index lists.
    score(d) = sum_m 1 / (k + rank_m(d)),  rank starts at 1."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, d in enumerate(ranking[:depth], start=1):
            scores[d] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class SearchEngine:
    METHODS = ["TF-IDF", "BM25", "Sentence Transformer", "Hybrid (RRF)"]

    def __init__(self, dense="auto"):
        """dense: 'sbert' | 'lsa' | 'auto' (sbert if available, else LSA fallback)."""
        self.doc_ids, self.titles, self.texts = load_docs()
        self.n = len(self.doc_ids)
        self.tokens = [preprocess(t) for t in self.texts]

        # Inverted index (term -> {doc_index: tf}) - shown in the report/demo
        self.inverted = defaultdict(dict)
        for i, toks in enumerate(self.tokens):
            for t in toks:
                self.inverted[t][i] = self.inverted[t].get(i, 0) + 1

        # TF-IDF
        self.tfidf = TfidfVectorizer(analyzer=lambda x: x)  # tokens already processed
        self.tfidf_mat = self.tfidf.fit_transform(self.tokens)

        # BM25
        self.bm25 = BM25Okapi(self.tokens)

        # Dense
        self._init_dense(dense)

    # ---- dense backend
    def _init_dense(self, dense):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.dense_name = None
        if dense in ("auto", "sbert"):
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(SBERT_MODEL)
                path = os.path.join(CACHE_DIR, "doc_emb_minilm.npy")
                if os.path.exists(path):
                    self.doc_emb = np.load(path)
                else:
                    self.doc_emb = self._model.encode(
                        self.texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
                    )
                    np.save(path, self.doc_emb)
                self._encode = lambda q: self._model.encode([q], normalize_embeddings=True)[0]
                self.dense_name = "Sentence Transformer (all-MiniLM-L6-v2)"
                return
            except Exception as e:  # noqa: BLE001
                if dense == "sbert":
                    raise
                print(f"[warn] sentence-transformers unavailable ({type(e).__name__}); using LSA fallback")
        # LSA fallback (NOT a transformer - only for testing where SBERT cannot be installed)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        svd = TruncatedSVD(n_components=200, random_state=0)
        self.doc_emb = normalize(svd.fit_transform(self.tfidf_mat))
        self._encode = lambda q: normalize(svd.transform(self.tfidf.transform([preprocess(q)])))[0]
        self.dense_name = "LSA fallback (NOT SBERT)"

    # ---- single rankers: each returns np.array of doc indices, best first
    def rank_tfidf(self, q):
        qv = self.tfidf.transform([preprocess(q)])
        s = (self.tfidf_mat @ qv.T).toarray().ravel()
        return np.argsort(-s, kind="stable")

    def rank_bm25(self, q):
        return np.argsort(-self.bm25.get_scores(preprocess(q)), kind="stable")

    def rank_dense(self, q):
        return np.argsort(-(self.doc_emb @ self._encode(q)), kind="stable")

    def rank_hybrid(self, q, k=60, depth=100):
        return np.array(
            rrf([self.rank_tfidf(q).tolist(), self.rank_bm25(q).tolist(), self.rank_dense(q).tolist()], k=k, depth=depth)
        )

    def rank(self, method, q):
        return {
            "TF-IDF": self.rank_tfidf,
            "BM25": self.rank_bm25,
            "Sentence Transformer": self.rank_dense,
            "Hybrid (RRF)": self.rank_hybrid,
        }[method](q)

    def to_doc_ids(self, idx):
        return [self.doc_ids[i] for i in idx]


# ---------------------------------------------------------------- 4. metrics
def precision_at_k(ranked, rel, k=10):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k=100):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def average_precision(ranked, rel):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def ndcg_at_k(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / np.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def interpolated_pr(ranked, rel, levels=np.linspace(0, 1, 11)):
    """11-point interpolated precision at standard recall levels."""
    hits, pts = 0, []
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            pts.append((hits / len(rel), hits / i))
    return np.array([max([p for r, p in pts if r >= lv], default=0.0) for lv in levels])
