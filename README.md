# NLP Project 2025
University Project in Python.

---
This project applies techniques of **semantic similarity**, **word embeddings**, and **automated text reconstruction** to transform unstructured or semantically unclear texts into clear, grammatically correct, and well-structured versions.

---
## Prerequisite tools

- Python 3.10+
- Poetry (Dependency Management)
- PyTorch & HuggingFace Transformers, TextAttack, Scikit-learn, NumPy / Pandas
- Internet connection for first installation
---

## Project Structure
``` 
src/
├── analysis/
│ ├── init.py
│ ├── embeddings.py # Word embedding models and utilities
│ ├── preprocess.py # Text preprocessing functions
│ └── visualize.py # Visualization (e.g. PCA, t-SNE)
│
├── util/
│ ├── pipelines/
│ │ ├── init.py
│ │ ├── custom_rewriter.py # Rule-based reconstruction
│ │ ├── huggingface_rewriter.py # T5-based paraphrasing
│ │ ├── similarity.py # Cosine similarity functions
│ │ └── textattack_rewriter.py # Paraphrasing using TextAttack
│ │
│ ├── init.py
│ └── texts.py # Original input texts
│
├── main.py # Main script to run the experiments
```
---

## How to Run

1) **Install dependencies:**

```bash
poetry install
```

2) **Execute the main script:**

```
poetry run python src/main.py
```



