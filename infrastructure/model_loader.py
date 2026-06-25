"""
Model Manager - Handles loading and managing all models with memory optimizations.
Contains one class: ModelManager.
"""

import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from scipy import sparse
from rank_bm25 import BM25Okapi
import json
import logging
import sys

import utils.text_utils as text_utils

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config):
        self.config = config
        
        self.bert_model = None
        self.bm25_model = None
        self.bm25_doc_ids = []
        
        self.embeddings_text_norm = None
        self.embeddings_title_norm = None
        self.doc_ids_list = []
        
        self.faiss_index_text = None
        self.faiss_index_title = None
        self.faiss_doc_ids = []
        
        self.tfidf_vectorizer = None
        self.tfidf_text_matrix = None
        self.tfidf_title_matrix = None
        
        self.bert_loaded = False
        self.bm25_loaded = False
        self.faiss_loaded = False
        self.tfidf_loaded = False
        
        self._load_all_models()
    
    def _load_all_models(self):
        logger.info("Starting to load all models immediately...")
        
        if self.config.LOAD_BM25_ON_START:
            self.load_bm25()
        if self.config.LOAD_FAISS_ON_START:
            self.load_faiss_indexes()
        if self.config.LOAD_TFIDF_ON_START:
            self.load_tfidf()
        if self.config.LOAD_BERT_ON_START:
            self.load_bert()
        
        logger.info("All models loaded successfully.")
    
    def load_bm25(self):
        try:
            model_path = self.config.MODELS_DIR / 'bm25_model.pkl'
            if not model_path.exists():
                logger.warning("BM25 model not found in models/")
                return False
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_model = data['bm25']
                self.bm25_doc_ids = data['doc_ids']
            self.bm25_loaded = True
            logger.info(f"BM25 model loaded ({len(self.bm25_doc_ids)} documents)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25: {e}")
            return False
    
    def load_faiss_indexes(self):
        try:
            faiss_text_path = self.config.MODELS_DIR / 'faiss_index_text.flat'
            faiss_title_path = self.config.MODELS_DIR / 'faiss_index_title.flat'
            faiss_ids_path = self.config.MODELS_DIR / 'faiss_doc_ids.pkl'
            
            if faiss_text_path.exists():
                self.faiss_index_text = faiss.read_index(str(faiss_text_path))
                logger.info("FAISS text index loaded")
            else:
                logger.info("FAISS text index not found")
            
            if faiss_title_path.exists():
                self.faiss_index_title = faiss.read_index(str(faiss_title_path))
                logger.info("FAISS title index loaded")
            else:
                logger.info("FAISS title index not found")
            
            if faiss_ids_path.exists():
                with open(faiss_ids_path, 'rb') as f:
                    self.faiss_doc_ids = pickle.load(f)
                logger.info(f"FAISS document IDs loaded ({len(self.faiss_doc_ids)} documents)")
            
            self.faiss_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS: {e}")
            return False
    
    def load_tfidf(self):
        try:
            vectorizer_path = self.config.MODELS_DIR / 'tfidf_vectorizer.pkl'
            text_matrix_path = self.config.MODELS_DIR / 'tfidf_text_matrix.npz'
            title_matrix_path = self.config.MODELS_DIR / 'tfidf_title_matrix.npz'
            
            if vectorizer_path.exists():
                main_module = sys.modules['__main__']
                funcs_to_import = ['identity', 'split_tokens', 'clean_query', 'normalize_scores']
                for func_name in funcs_to_import:
                    if not hasattr(main_module, func_name):
                        if hasattr(text_utils, func_name):
                            setattr(main_module, func_name, getattr(text_utils, func_name))
                            logger.info(f"Function {func_name} temporarily defined in __main__ for TF-IDF loading")
                        else:
                            if func_name == 'identity':
                                def identity(x): return x
                                setattr(main_module, 'identity', identity)
                            elif func_name == 'split_tokens':
                                def split_tokens(x): return x.split()
                                setattr(main_module, 'split_tokens', split_tokens)
                
                with open(vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("TF-IDF Vectorizer loaded")
            
            if text_matrix_path.exists():
                matrix = sparse.load_npz(text_matrix_path)
                if matrix.dtype != np.float32:
                    matrix = matrix.astype(np.float32)
                self.tfidf_text_matrix = matrix
                logger.info(f"TF-IDF Text Matrix loaded ({matrix.shape} - {matrix.dtype})")
            
            if title_matrix_path.exists():
                matrix = sparse.load_npz(title_matrix_path)
                if matrix.dtype != np.float32:
                    matrix = matrix.astype(np.float32)
                self.tfidf_title_matrix = matrix
                logger.info(f"TF-IDF Title Matrix loaded ({matrix.shape} - {matrix.dtype})")
            
            self.tfidf_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load TF-IDF: {e}")
            return False
    
    def load_bert(self):
        try:
            if self.bert_model is None:
                local_model_path = self.config.MODELS_DIR / 'bert_model'
                if local_model_path.exists():
                    self.bert_model = SentenceTransformer(str(local_model_path))
                    logger.info("BERT loaded from local folder")
                else:
                    logger.warning("Local model not found, downloading from Hugging Face...")
                    self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.bert_model.save(str(local_model_path))
                    logger.info("BERT downloaded and saved locally")
            
            text_norm_path = self.config.MODELS_DIR / 'embeddings_text_norm.npy'
            title_norm_path = self.config.MODELS_DIR / 'embeddings_title_norm.npy'
            
            if not text_norm_path.exists():
                raw_path = self.config.MODELS_DIR / 'embeddings_text.npy'
                if raw_path.exists():
                    logger.info("Computing normalized text embeddings...")
                    raw = np.load(raw_path, mmap_mode='r')
                    norms = np.linalg.norm(raw, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    norm_emb = raw / norms
                    np.save(text_norm_path, norm_emb.astype(np.float32))
                    logger.info(f"Normalized text embeddings saved to {text_norm_path}")
                    del raw, norms, norm_emb
                else:
                    raise FileNotFoundError('embeddings_text.npy not found')
            
            if not title_norm_path.exists():
                raw_path = self.config.MODELS_DIR / 'embeddings_title.npy'
                if raw_path.exists():
                    logger.info("Computing normalized title embeddings...")
                    raw = np.load(raw_path, mmap_mode='r')
                    norms = np.linalg.norm(raw, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    norm_emb = raw / norms
                    np.save(title_norm_path, norm_emb.astype(np.float32))
                    logger.info(f"Normalized title embeddings saved to {title_norm_path}")
                    del raw, norms, norm_emb
                else:
                    logger.warning("embeddings_title.npy not found, using text embeddings only")
            
            if text_norm_path.exists():
                self.embeddings_text_norm = np.load(text_norm_path, mmap_mode='r')
                logger.info("Normalized text embeddings loaded (mmap - RAM efficient)")
            else:
                raise FileNotFoundError('embeddings_text_norm.npy not found')
            
            if title_norm_path.exists():
                self.embeddings_title_norm = np.load(title_norm_path, mmap_mode='r')
                logger.info("Normalized title embeddings loaded (mmap - RAM efficient)")
            else:
                self.embeddings_title_norm = None
                logger.warning("Title embeddings not available")
            
            if not self.doc_ids_list:
                self.doc_ids_list = self._load_doc_ids()
            
            self.bert_loaded = True
            logger.info("BERT fully loaded (with mmap memory optimization)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BERT: {e}")
            return False
    
    def _load_doc_ids(self):
        doc_path = self.config.DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
        if doc_path.exists():
            doc_ids = []
            with open(doc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    doc_id = data.get('doc_id')
                    if doc_id:
                        doc_ids.append(str(doc_id))
            if doc_ids:
                logger.info(f"Loaded {len(doc_ids)} document IDs")
                return doc_ids
        
        for candidate in [self.config.MODELS_DIR / 'doc_ids.json',
                          self.config.MODELS_DIR / 'doc_ids.pkl',
                          self.config.MODELS_DIR / 'doc_ids.txt']:
            if candidate.exists():
                if candidate.suffix == '.json':
                    with open(candidate, 'r') as f:
                        return json.load(f)
                if candidate.suffix == '.pkl':
                    with open(candidate, 'rb') as f:
                        return pickle.load(f)
                with open(candidate, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
        
        logger.warning("No real document IDs found")
        return []
    
    def get_bert(self):
        return self.bert_model
    
    def get_bm25(self):
        return self.bm25_model
    
    def get_faiss_text(self):
        return self.faiss_index_text
    
    def get_faiss_title(self):
        return self.faiss_index_title