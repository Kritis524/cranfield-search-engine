"""Evaluate all four rankers on Cranfield -> results/comparison_table.csv + pr_curve.png"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ir_core import (SearchEngine, average_precision, interpolated_pr, load_qrels,
                     load_queries, ndcg_at_k, precision_at_k, recall_at_k)

ap = argparse.ArgumentParser()
ap.add_argument("--dense", default="auto", choices=["auto", "sbert", "lsa"])
args = ap.parse_args()

os.makedirs("results", exist_ok=True)
engine = SearchEngine(dense=args.dense)
queries, qrels = load_queries(), load_qrels()
qids = [q for q in queries if q in qrels]
print(f"Dense model: {engine.dense_name} | evaluating {len(qids)} queries")

rows, pr_curves = [], {}
levels = np.linspace(0, 1, 11)
for m in engine.METHODS:
    P, R, AP, N, PR = [], [], [], [], []
    for q in qids:
        ranked = engine.to_doc_ids(engine.rank(m, queries[q]))
        rel = qrels[q]
        P.append(precision_at_k(ranked, rel, 10))
        R.append(recall_at_k(ranked, rel, 100))
        AP.append(average_precision(ranked, rel))
        N.append(ndcg_at_k(ranked, rel, 10))
        PR.append(interpolated_pr(ranked, rel))
    label = m + " [LSA fallback]" if "fallback" in engine.dense_name and m in ("Sentence Transformer", "Hybrid (RRF)") else m
    rows.append({"Method": label, "Precision@10": np.mean(P), "Recall@100": np.mean(R),
                 "MAP": np.mean(AP), "nDCG@10": np.mean(N)})
    pr_curves[m] = np.mean(PR, axis=0)

df = pd.DataFrame(rows).set_index("Method").round(4)
df.to_csv("results/comparison_table.csv")
print(df.to_string())

plt.figure(figsize=(7, 5))
for m, c in pr_curves.items():
    plt.plot(levels, c, marker="o", label=m)
plt.xlabel("Recall"); plt.ylabel("Precision (interpolated)")
plt.title(f"Precision-Recall curves - Cranfield ({len(qids)} queries)")
plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig("results/pr_curve.png", dpi=200)
print("saved results/comparison_table.csv and results/pr_curve.png")
