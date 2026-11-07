"""Streamlit search interface:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from ir_core import SearchEngine, load_qrels, load_queries

st.set_page_config(page_title="Cranfield Search Engine", layout="wide")


@st.cache_resource(show_spinner="Building indexes (first run encodes 1,400 docs)...")
def get_engine():
    return SearchEngine(dense="auto")


engine = get_engine()
queries, qrels = load_queries(), load_qrels()

st.title("Cranfield Search Engine")
st.caption(f"1,400 aeronautical abstracts  |  Dense model: {engine.dense_name}")

tab_search, tab_eval = st.tabs(["Search", "Evaluation results"])

with tab_search:
    sample = st.selectbox("Pick a benchmark query (optional)",
                          [""] + [f"{q}: {t}" for q, t in queries.items() if q in qrels])
    default_q = sample.split(": ", 1)[1] if sample else ""
    query = st.text_input("Search query", value=default_q,
                          placeholder="e.g. boundary layer transition at supersonic speeds")
    top_k = st.slider("Results per method", 5, 20, 10)

    if query.strip():
        rel = qrels.get(int(sample.split(":")[0]), {}) if sample else {}
        cols = st.columns(4)
        for col, m in zip(cols, engine.METHODS):
            with col:
                st.subheader(m)
                for rank, idx in enumerate(engine.rank(m, query)[:top_k], start=1):
                    did = engine.doc_ids[idx]
                    mark = " ✅" if did in rel else ""
                    st.markdown(f"**{rank}. {engine.titles[idx][:90]}**{mark}  \n"
                                f"<span style='color:gray;font-size:12px'>doc {did}</span>",
                                unsafe_allow_html=True)
        if rel:
            st.info(f"✅ = judged relevant in Cranfield qrels ({len(rel)} relevant docs for this query)")

with tab_eval:
    if os.path.exists("results/comparison_table.csv"):
        st.dataframe(pd.read_csv("results/comparison_table.csv"), hide_index=True)
        st.image("results/pr_curve.png")
    else:
        st.warning("Run `python evaluate.py` first.")
