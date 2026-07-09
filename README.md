# Distill

**A catalogue intelligence system that finds duplicate, near-duplicate, and mislabeled product listings in large e-commerce catalogues — and uses a retrieval-augmented agent to generate clean, corrected entries.**

---

## The Problem

Fast-growing e-commerce and quick-commerce catalogues accumulate noise: near-identical SKUs listed under slightly different names ("Amul Butter 500g" vs. "Amul Butter — 500 gm Pack"), inconsistent categorization, missing attributes, and conflicting prices for what is effectively the same product.

This isn't just untidy data — it has real operational cost:
- **Fragmented demand signal.** When one product exists as three separate SKUs, inventory and demand-forecasting systems see three weak signals instead of one strong one.
- **Degraded search & discovery.** Customers see cluttered, redundant results instead of a single clear listing.
- **Manual cleanup overhead.** Catalogue teams spend hours doing what a similarity-based system can flag automatically.

Distill tackles this as an **entity resolution problem**: given a large, messy catalogue, identify which listings refer to the same real-world product, and produce a single canonical, corrected entry for each cluster.

---

## How It Works

**1. Duplicate & Near-Duplicate Detection**
Product titles and descriptions are embedded using `sentence-transformers`. Embeddings are indexed with FAISS for approximate nearest-neighbor search, and candidate listings are clustered by semantic similarity — catching cases that simple string-matching would miss (e.g. reordered words, abbreviations, unit formatting differences).

**2. Attribute & Price Consistency Checks**
A rule-based layer inspects each candidate cluster for inconsistencies: category mismatches, missing attributes (size, brand, pack count), and price outliers relative to the rest of the cluster (via z-score).

**3. Retrieval-Augmented Correction**
For each flagged cluster, a RAG pipeline retrieves the most complete/canonical listing as reference context, then an LLM generates:
- A single consolidated, corrected listing
- A short natural-language justification for the merge/correction
- A confidence flag for cases needing human review

**4. Evaluation**
Synthetic duplicates are injected into a real catalogue dataset to create ground truth, allowing precision/recall/F1 measurement on duplicate detection — keeping evaluation rigorous even though the underlying use case is real-world.

**5. Interactive Demo**
A Streamlit app lets you select or upload a catalogue slice and see flagged clusters plus agent-generated corrections in real time.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data handling | pandas |
| Embeddings | sentence-transformers |
| Similarity search | FAISS |
| Vector store (RAG) | ChromaDB |
| Generation | Groq API (`qwen/qwen3.6-27b`, fallback `openai/gpt-oss-120b`) |
| Evaluation | scikit-learn |
| Demo | Streamlit |
| Model hosting | HuggingFace Hub |



---

## Results

*(To be filled in once benchmarking is complete)*

| Metric | Score |
|---|---|
| Duplicate detection precision | — |
| Duplicate detection recall | — |
| Duplicate detection F1 | — |
| Attribute mismatch false-positive rate | — |

---

## Dataset

This project uses a public grocery/e-commerce product dataset as a stand-in for a real quick-commerce catalogue. Synthetic duplicate and mislabeling noise is injected into a clean subset to construct ground truth for evaluation. No proprietary or company-specific data is used.

---

## Setup

```bash
git clone https://github.com/<your-username>/distill.git
cd distill
pip install -r requirements.txt
```

Configure your API keys in `.env`:
```
GROQ_API_KEY=your_key_here
```

Run the demo:
```bash
streamlit run app.py
```

---

distill/
├── .github/
│   └── workflows/
│       └── ci.yml                  # lint + test on push
│
├── configs/
│   ├── config.yaml                 # thresholds, model names, paths
│   └── .env.example                # API key template
│
├── data/
│   ├── raw/                        # original catalogue dataset (untouched)
│   ├── processed/                  # cleaned + noise-injected catalogue
│   └── ground_truth/               # synthetic duplicate labels for eval
│
├── src/
│   └── distill/
│       ├── __init__.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── loader.py           # dataset loading + validation
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── encoder.py          # sentence-transformer wrapper
│       │   └── index.py            # FAISS index build/query
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── clustering.py       # duplicate/near-duplicate clustering
│       │   └── rules.py            # attribute + price mismatch checks
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── retriever.py        # ChromaDB retrieval logic
│       │   ├── generator.py        # Groq LLM correction + justification
│       │   └── prompts.py          # prompt templates, versioned
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py          # precision/recall/F1 scoring
│       └── pipeline.py             # end-to-end orchestration
│
├── app/
│   ├── streamlit_app.py            # interactive demo
│   └── assets/                     # demo screenshots, logo, GIF
│
├── notebooks/
│   ├── 01_eda.ipynb                # catalogue exploration
│   ├── 02_embedding_tuning.ipynb   # similarity threshold tuning
│   └── 03_evaluation.ipynb         # metric walkthroughs, error analysis
│
├── tests/
│   ├── test_clustering.py
│   ├── test_rules.py
│   └── test_pipeline.py
│
├── docs/
│   ├── architecture.md             # system design + diagram
│   └── results.md                  # benchmark writeup, sample outputs
│
├── scripts/
│   ├── build_index.py              # one-shot: catalogue → FAISS index
│   └── run_evaluation.py           # one-shot: reproduce benchmark numbers
│
├── .gitignore
├── LICENSE
├── pyproject.toml                  # packaging + dependency management
├── requirements.txt
└── README.md

---

## Why This Exists

Catalogue quality is one of those "unglamorous layer" problems — not as flashy as a recommendation engine, but it's the foundation everything else (search, forecasting, pricing) depends on being accurate. Distill is built to show that entity resolution + retrieval-augmented generation can turn a manual cleanup task into an automated, explainable pipeline.
