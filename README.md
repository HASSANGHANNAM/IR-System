# 🔍 IR-System - Information Retrieval Search Engine

A modular, high-performance **Information Retrieval (IR) System** built with Python and Flask.  
Supports multiple search models (TF‑IDF, BM25, BERT, Hybrid) with evaluation metrics, interactive charts, and a clean user interface.

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the System](#-running-the-system)
- [API Endpoints](#-api-endpoints)
- [Charts & Evaluation](#-charts--evaluation)
- [Authors](#-authors)

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Multiple Search Models** | TF‑IDF, BM25, BERT (with FAISS acceleration), and Hybrid (Parallel/Serial) |
| **Query Refinement** | Spell-checking and duplicate removal for cleaner queries |
| **Evaluation Metrics** | Precision@k, MAP, NDCG, Recall@k for performance assessment |
| **Interactive Charts** | Model comparison, BM25 parameter tuning, Hybrid weight experiments, Serial permutations |
| **User Interface** | Clean frontend (HTML/CSS/JS) with real-time search and results display |
| **Caching (LRU)** | Speeds up repeated queries with intelligent caching |
| **Service-Oriented Architecture (SOA)** | Separation of concerns: Routes → Services → Infrastructure |
| **Memory Optimized** | Uses `mmap` for embeddings and `float32` for sparse matrices |

---

## 🧰 Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.14, Flask, Werkzeug |
| **Search Models** | Sentence‑Transformers (BERT), FAISS, Rank‑BM25, Scikit‑learn |
| **NLP** | NLTK (tokenization, stopwords), pyspellchecker |
| **Data Processing** | NumPy, SciPy |
| **Database** | SQLite |
| **Frontend** | HTML5, CSS3, JavaScript (Chart.js for charts) |
| **Version Control** | Git |

---

## 📂 Project Structure
.
├── archives
│   ├── additional_backup.zip
│   ├── ir_datasets_backup.zip
│   ├── processed_lemmatization.zip
│   ├── processed_stemming.zip
│   └── tfidf_models.zip
├── assets
│   └── Screenshot_٢٠٢٦٠٦٢٠_٠٣٤٦٥٥.png
├── core
│   ├── __init__.py
│   └── models.py
├── data
│   ├── processed_lemmatization
│   │   └── content
│   ├── processed_stemming
│   │   ├── corpus_cleaned_text.jsonl
│   │   ├── corpus_cleaned_title.jsonl
│   │   ├── corpus_original.jsonl
│   │   └── queries_cleaned.jsonl
│   ├── raw
│   │   ├── ir_data.db
│   │   ├── ir.txt
│   │   └── q.txt
│   ├── results
│   │   ├── bm25_params_results.json
│   │   ├── comparison_results.json
│   │   ├── hybrid_full_results.json
│   │   ├── refinement_impact_results.json
│   │   ├── refinement_impact_spellcheck_only.json
│   │   └── serial_permutations_results.json
│   ├── ir_data.db
│   └── test.qrels
├── infrastructure
│   ├── cache_manager.py
│   ├── database.py
│   ├── db_initializer.py
│   ├── file_handlers.py
│   ├── __init__.py
│   └── model_loader.py
├── models
│   ├── bert_model
│   │   ├── 1_Pooling
│   │   ├── 2_Normalize
│   │   ├── config.json
│   │   ├── config_sentence_transformers.json
│   │   ├── model.safetensors
│   │   ├── modules.json
│   │   ├── README.md
│   │   ├── sentence_bert_config.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── bm25_model.pkl
│   ├── embeddings_text_norm.npy
│   ├── embeddings_text.npy
│   ├── embeddings_title_norm.npy
│   ├── embeddings_title.npy
│   ├── faiss_doc_ids.pkl
│   ├── faiss_index_text.flat
│   ├── faiss_index_title.flat
│   ├── tfidf_feature_names.pkl
│   ├── tfidf_text_matrix_backup.npz
│   ├── tfidf_text_matrix.npz
│   ├── tfidf_title_matrix.npz
│   └── tfidf_vectorizer.pkl
├── notebooks
│   ├── 123.ipynb
│   ├── 1.ipynb
│   ├── END'.ipynb
│   ├── END.ipynb
│   ├── ir.ipynb
│   ├── Untitled0.ipynb
│   ├── Untitled1.ipynb
│   ├── Untitled2.ipynb
│   ├── Untitled3.ipynb
│   └── Untitled4.ipynb
├── routes
│   ├── admin_routes.py
│   ├── chart_routes.py
│   ├── __init__.py
│   └── search_routes.py
├── scripts
│   ├── build_faiss_index.py
│   ├── convert_tfidf_to_float16.py
│   ├── save_vectorizer_local.py
│   └── setup_local.py
├── services
│   ├── __init__.py
│   ├── ranking_service.py
│   └── retrieval_service.py
├── tests
│   ├── unit
│   │   ├── compare_models.py
│   │   ├── __init__.py
│   │   ├── test_bm25_params.py
│   │   ├── test_hybrid_weights.py
│   │   ├── test_refinement_impact.py
│   │   ├── test_serial_all_permutations.py
│   │   ├── test_speed.py
│   │   └── test_spellcheck_queries.py
│   └── __init__.py
├── UI
│   ├── index.html
│   ├── script.js
│   └── style.css
├── utils
│   ├── __init__.py
│   └── text_utils.py
├── app_factory.py
├── app.py
├── config.py
├── IR-System.postman_collection.json
├── README.md
└── requirements.txt


---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/IR-System.git
cd IR-System


2️⃣ Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate        # Windows


3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Download NLTK Data (if required)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

5️⃣ Prepare Data & Models
Ensure the following directories exist and contain the required files:

data/processed_stemming/ – corpus_original.jsonl, queries_cleaned.jsonl

models/ – trained model files (BM25, FAISS, BERT, TF‑IDF)

If models are missing, the system will attempt to load them from local paths as configured in config.py.


▶️ Running the System

python app.py
Access the Application
Web Interface: http://127.0.0.1:5000

Logs
All application logs are written to logs/app.log.
Server startup messages (e.g., * Running on...) appear in the terminal.

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/search` | Execute a search. Supports `model` (`tfidf`, `bm25`, `bert`, `hybrid`), `top_k`, `refine`, `evaluate`, `use_faiss`. |
| `GET` | `/api/queries` | Fetch all 49 predefined queries (with IDs). |
| `GET` | `/api/document/<doc_id>` | Retrieve a document by its ID (title + full text). |
| `GET` | `/api/admin/status` | System status: loaded models, cache size, document count. |
| `POST` | `/api/admin/cache/clear` | Clear the query cache (all or specific type). |
| `POST` | `/api/admin/models/load/<model_name>` | Dynamically load a model (`bm25`, `faiss`, `tfidf`, `bert`). |
| `POST` | `/api/admin/models/unload` | Unload all models from memory. |


## 📊 Charts & Evaluation
| Chart | File | Purpose |
| :--- | :--- | :--- |
| **Model Comparison** | `comparison_results.json` | Compare TF‑IDF, BM25, BERT, Hybrid |
| **BM25 Parameters** | `bm25_params_results.json` | Tune `k1` and `b` for BM25 |
| **Hybrid Weights** | `hybrid_full_results.json` | Test different weight combinations (TF‑IDF / BM25 / BERT) |
| **Serial Permutations** | `serial_permutations_results.json` | Evaluate different model sequences in serial mode |



👥 Authors
HASSAN GHANAM
ghaith abo rashed 
abd alrahman houir
alaa suliman 

