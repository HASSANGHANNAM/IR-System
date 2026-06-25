from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
UI_DIR = ROOT / 'UI'
ARCHIVES_DIR = ROOT / 'archives'
ASSETS_DIR = ROOT / 'assets'

DOCUMENTS_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
QUERIES_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'queries_cleaned.jsonl'
DB_PATH = DATA_DIR / 'ir_data.db'
QRELS_PATHS = [
    DATA_DIR / 'test.qrels',
    DATA_DIR / 'beir' / 'webis-touche2020' / 'test.qrels',
    ROOT / 'beir' / 'webis-touche2020' / 'test.qrels',
]

BM25_MODEL_PATH = MODELS_DIR / 'bm25_model.pkl'
FAISS_TEXT_INDEX_PATH = MODELS_DIR / 'faiss_index_text.flat'
FAISS_TITLE_INDEX_PATH = MODELS_DIR / 'faiss_index_title.flat'
FAISS_DOC_IDS_PATH = MODELS_DIR / 'faiss_doc_ids.pkl'

TFIDF_VECTORIZER_PATH = MODELS_DIR / 'tfidf_vectorizer.pkl'
TFIDF_TEXT_MATRIX_PATH = MODELS_DIR / 'tfidf_text_matrix.npz'
TFIDF_TITLE_MATRIX_PATH = MODELS_DIR / 'tfidf_title_matrix.npz'
TFIDF_TEXT_MATRIX_BACKUP_PATH = MODELS_DIR / 'tfidf_text_matrix_backup.npz'

BERT_MODEL_PATH = MODELS_DIR / 'bert_model'
EMBEDDINGS_TEXT_PATH = MODELS_DIR / 'embeddings_text.npy'
EMBEDDINGS_TITLE_PATH = MODELS_DIR / 'embeddings_title.npy'

LOAD_BM25_ON_START = True
LOAD_FAISS_ON_START = True
LOAD_TFIDF_ON_START = True
LOAD_BERT_ON_START = True

DEFAULT_TOP_K = 10
MAX_TOP_K = 100
HYBRID_MODE_DEFAULT = 'parallel'

HYBRID_WEIGHT_TFIDF = 0.3
HYBRID_WEIGHT_BM25 = 0.3
HYBRID_WEIGHT_BERT = 0.4

EVALUATION_TOP_K = 10
QRELS_SCORE_THRESHOLD = 0

BM25_K1_DEFAULT = 2.0
BM25_B_DEFAULT = 0.60

ENABLE_QUERY_CACHE = True
QUERY_CACHE_SIZE = 1000
CACHE_ALGORITHM = 'lru'

# ========== Logging Settings ==========
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ✅ Log file path
LOGS_DIR = ROOT / 'logs'
LOG_FILE_PATH = LOGS_DIR / 'app.log'