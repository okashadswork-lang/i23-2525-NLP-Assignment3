# RAG Pipeline — Transformer-Based Review Understanding

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built entirely from scratch for CS-4063: Natural Language Processing (FAST NUCES). The system extracts structured information from Amazon product reviews, retrieves semantically similar examples, and generates natural language explanations — all using custom Transformer architectures with no pretrained models.

---

## 🏗️ System Architecture

```
Amazon Reviews
      │
      ▼
┌─────────────────┐
│   Preprocessing  │  Text cleaning → Tokenization → Vocab → Padding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Part A: Encoder │  Encoder-only Transformer (from scratch)
│  (Multi-task)    │  → Sentiment Classification (Neg / Neu / Pos)
│                  │  → Derived Feature Prediction
│                  │  → Review Embeddings (saved to disk)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Part B: Retrieval│  Cosine similarity search over training embeddings
│  Module          │  → Top-k similar reviews retrieved at inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Part C: Decoder │  Decoder-only Transformer (from scratch)
│  (RAG Generator) │  Input: review + sentiment + feature + retrieved context
│                  │  → Autoregressive explanation generation (1–2 sentences)
└─────────────────┘
```

---

## 📦 Dataset

- **Source:** Amazon Reviews Dataset — [nijianmo.github.io/amazon](https://nijianmo.github.io/amazon/index.html)
- **Size:** 30,000–45,000 reviews across 3+ product categories (~10–15k per category)
- **Labels:** Star ratings 1–5 mapped to sentiment: Negative (1–2), Neutral (3), Positive (4–5)
- **Split:** 70% Train / 15% Validation / 15% Test

---

## 🔧 Parts

### Part A — Encoder Model (25 marks)
- Encoder-only Transformer implemented from scratch
- Multi-task learning: sentiment classification + derived feature prediction
- Custom attention mechanism (no `nn.MultiheadAttention`)
- Multi-head attention, residual connections, layer normalization, feed-forward blocks
- Review embeddings saved for retrieval

### Part B — Retrieval Module (15 marks)
- All training embeddings stored and indexed
- Cosine similarity search at inference time
- Configurable top-k retrieval
- Retrieval quality analysis with query/result examples

### Part C — Decoder Model (25 marks)
- Decoder-only Transformer implemented from scratch
- Causal (masked) self-attention — cannot attend to future tokens
- Input: review text + predicted sentiment + derived feature + top-k retrieved reviews
- Autoregressive token-by-token generation
- Evaluated with perplexity on test set
- RAG ablation study: full system vs. no-retrieval baseline

---

## 🚫 Restrictions

Built without any of the following:
- Pretrained models (BERT, GPT, T5, etc.)
- `nn.Transformer`
- `nn.MultiheadAttention`
- `nn.TransformerEncoder`

Everything is implemented from scratch using PyTorch primitives.

---

## 📁 Directory Structure

```
rag-pipeline-nlp/
├── i23XXXX-NLP-Assignment2.ipynb   # Main notebook (runnable top to bottom)
├── models/                          # Saved model weights
├── results/                         # Embeddings, metrics, evaluation outputs
└── README.md
```

---

## ▶️ How to Run

1. Clone the repo
```bash
git clone https://github.com/okashadswork-lang/rag-pipeline-nlp.git
cd rag-pipeline-nlp
```

2. Install dependencies
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

3. Download the Amazon Reviews dataset from [here](https://nijianmo.github.io/amazon/index.html) and place CSVs in the project root

4. Open and run the notebook top to bottom
```bash
jupyter notebook i23XXXX-NLP-Assignment2.ipynb
```

---

## 📈 Evaluation

| Component | Metric |
|-----------|--------|
| Sentiment Classification | Accuracy, F1 |
| Derived Feature | Task-specific metric |
| Retrieval Quality | Semantic similarity analysis |
| Generation | Perplexity on test set |
| RAG Ablation | Full system vs. no-retrieval baseline |

---

## 🛠️ Tech Stack

- **Framework:** PyTorch
- **Language:** Python
- **Notebook:** Jupyter

---

## 👤 Author

**Okasha Yamin** — [@okashadswork-lang](https://github.com/okashadswork-lang)  
CS-4063: Natural Language Processing — FAST NUCES
