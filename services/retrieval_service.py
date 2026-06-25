"""
Retrieval Service - Handles all search algorithms.
Contains one class: RetrievalService.
"""

import numpy as np
import faiss
import logging
from typing import List, Dict

from utils.text_utils import clean_query, normalize_scores

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, config, model_manager, db_manager, cache_manager):
        self.config = config
        self.models = model_manager
        self.db = db_manager
        self.cache = cache_manager
    
    # ========== TF-IDF Search ==========
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Dict]:
        vectorizer = self.models.tfidf_vectorizer
        text_matrix = self.models.tfidf_text_matrix
        doc_ids = self.models.doc_ids_list
        
        if vectorizer is None or text_matrix is None:
            logger.error("TF-IDF models not loaded")
            return []
        
        cleaned_query = clean_query(query)
        query_vec = vectorizer.transform([cleaned_query])
        scores = query_vec.dot(text_matrix.T).toarray().flatten()
        
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                doc_id = doc_ids[idx] if idx < len(doc_ids) else str(idx)
                results.append({'doc_id': doc_id, 'score': float(score)})
        return results
    
    # ========== BM25 Search ==========
    def search_bm25(self, query: str, top_k: int = 10, k1: float = 2.0, b: float = 0.60) -> List[Dict]:
        bm25_model = self.models.get_bm25()
        doc_ids = self.models.bm25_doc_ids
        
        if bm25_model is None:
            logger.error("BM25 model not loaded")
            return []
        
        if k1 != 2.0:
            bm25_model.k1 = k1
        if b != 0.60:
            bm25_model.b = b
        
        cleaned_query = clean_query(query)
        tokenized_query = cleaned_query.split()
        scores = bm25_model.get_scores(tokenized_query)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if idx < len(doc_ids):
                results.append({
                    'doc_id': doc_ids[idx],
                    'score': float(scores[idx])
                })
        return results
    
    # ========== BERT Search (with optional FAISS) ==========
    def search_bert(self, query: str, top_k: int = 10, use_title: bool = True,
                    alpha: float = 0.4, use_faiss: bool = False) -> List[Dict]:
        bert_model = self.models.get_bert()
        embeddings_text = self.models.embeddings_text_norm
        embeddings_title = self.models.embeddings_title_norm
        doc_ids = self.models.doc_ids_list
        
        if bert_model is None or embeddings_text is None:
            logger.error("BERT not loaded")
            return []
        
        cleaned_query = clean_query(query)
        query_vec = bert_model.encode([cleaned_query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm != 0:
            query_vec = query_vec / query_norm
        
        # FAISS accelerated search
        if use_faiss:
            faiss_text = self.models.get_faiss_text()
            faiss_title = self.models.get_faiss_title()
            faiss_doc_ids = self.models.faiss_doc_ids
            
            if faiss_text is not None:
                query_vec_faiss = query_vec.astype(np.float32).reshape(1, -1)
                faiss.normalize_L2(query_vec_faiss)
                
                k_faiss = max(top_k * 2, 50)
                if use_title and faiss_title is not None:
                    distances, indices = faiss_title.search(query_vec_faiss, k_faiss)
                else:
                    distances, indices = faiss_text.search(query_vec_faiss, k_faiss)
                
                top_indices = indices[0][:top_k]
                scores = distances[0][:top_k]
                
                results = []
                for idx, score in zip(top_indices, scores):
                    if idx < len(faiss_doc_ids):
                        doc_id = faiss_doc_ids[idx]
                    else:
                        doc_id = doc_ids[idx] if idx < len(doc_ids) else str(idx)
                    results.append({'doc_id': str(doc_id), 'score': float(score)})
                return results
        
        # Brute-force search
        if use_title and embeddings_title is not None:
            title_scores = np.dot(query_vec, embeddings_title.T)
            text_scores = np.dot(query_vec, embeddings_text.T)
            scores = alpha * title_scores + (1.0 - alpha) * text_scores
        else:
            scores = np.dot(query_vec, embeddings_text.T)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                'doc_id': doc_ids[idx] if idx < len(doc_ids) else str(idx),
                'score': float(scores[idx]),
            }
            for idx in top_indices
            if scores[idx] > 0
        ]
    
    # ========== Hybrid Search ==========
    def search_hybrid(self, query: str, hybrid_mode: str, weights: Dict[str, float],
                      top_k: int = 10, use_faiss: bool = False) -> List[Dict]:
        doc_ids = self.models.doc_ids_list
        total_docs = len(doc_ids)
        
        if hybrid_mode == 'parallel':
            if use_faiss and weights.get('bert', 0) > 0 and self.models.get_faiss_text() is not None:
                candidates_k = min(top_k * 3, total_docs, 200)
                bert_results = self.search_bert(query, top_k=candidates_k, use_title=True, alpha=0.4, use_faiss=True)
                candidate_doc_ids = [item['doc_id'] for item in bert_results]
                
                if not candidate_doc_ids:
                    return []
                
                tfidf_scores_all = self._get_all_tfidf_scores(query)
                bm25_scores_all = self._get_all_bm25_scores(query) if weights.get('bm25', 0) > 0 else None
                
                results_dict = {}
                for doc_id in candidate_doc_ids:
                    try:
                        idx = doc_ids.index(doc_id)
                    except ValueError:
                        continue
                    
                    scores = {}
                    if weights.get('tfidf', 0) > 0:
                        scores['tfidf'] = tfidf_scores_all[idx]
                    if weights.get('bm25', 0) > 0 and bm25_scores_all is not None:
                        scores['bm25'] = bm25_scores_all[idx]
                    if weights.get('bert', 0) > 0:
                        bert_score = next((item['score'] for item in bert_results if item['doc_id'] == doc_id), 0.0)
                        scores['bert'] = bert_score
                    
                    final_score = sum(w * scores.get(name, 0.0) for name, w in weights.items())
                    results_dict[doc_id] = final_score
                
                sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
                return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in sorted_results]
            
            else:
                scores_dict = {}
                if weights.get('tfidf', 0) > 0:
                    scores_dict['tfidf'] = self._get_all_tfidf_scores(query)
                if weights.get('bm25', 0) > 0:
                    scores_dict['bm25'] = self._get_all_bm25_scores(query)
                if weights.get('bert', 0) > 0:
                    scores_dict['bert'] = self._get_all_bert_scores(query)
                
                if not scores_dict:
                    raise ValueError('At least one model must have weight > 0')
                
                norm_scores = {name: normalize_scores(scores) for name, scores in scores_dict.items()}
                final_scores = np.zeros(total_docs)
                for name, scores in norm_scores.items():
                    final_scores += weights[name] * scores
                
                top_indices = np.argsort(final_scores)[::-1][:top_k]
                return [
                    {'doc_id': doc_ids[idx], 'score': float(final_scores[idx])}
                    for idx in top_indices
                ]
        
        elif hybrid_mode == 'serial':
            tfidf_scores = self._get_all_tfidf_scores(query)
            initial_k = min(100, total_docs)
            top_tfidf_indices = np.argsort(tfidf_scores)[::-1][:initial_k]
            
            if len(top_tfidf_indices) == 0:
                return []
            
            all_bm25_scores = self._get_all_bm25_scores(query)
            bm25_scores_subset = all_bm25_scores[top_tfidf_indices]
            reranked_order = np.argsort(bm25_scores_subset)[::-1][:top_k]
            
            final_indices = top_tfidf_indices[reranked_order]
            final_scores = bm25_scores_subset[reranked_order]
            
            return [
                {'doc_id': doc_ids[idx], 'score': float(score)}
                for idx, score in zip(final_indices, final_scores)
            ]
        else:
            raise ValueError("Invalid hybrid_mode. Must be 'parallel' or 'serial'")
    
    # ========== Helper methods for all scores ==========
    def _get_all_tfidf_scores(self, query: str) -> np.ndarray:
        vectorizer = self.models.tfidf_vectorizer
        text_matrix = self.models.tfidf_text_matrix
        if vectorizer is None or text_matrix is None:
            logger.error("TF-IDF models not loaded")
            return np.array([])
        cleaned_query = clean_query(query)
        query_vec = vectorizer.transform([cleaned_query])
        return query_vec.dot(text_matrix.T).toarray().flatten()
    
    def _get_all_bm25_scores(self, query: str) -> np.ndarray:
        bm25_model = self.models.get_bm25()
        if bm25_model is None:
            logger.error("BM25 model not loaded")
            return np.array([])
        cleaned_query = clean_query(query)
        tokenized_query = cleaned_query.split()
        return np.array(bm25_model.get_scores(tokenized_query))
    
    def _get_all_bert_scores(self, query: str) -> np.ndarray:
        bert_model = self.models.get_bert()
        embeddings_text = self.models.embeddings_text_norm
        if bert_model is None or embeddings_text is None:
            logger.error("BERT not loaded")
            return np.array([])
        cleaned_query = clean_query(query)
        query_vec = bert_model.encode([cleaned_query])[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm != 0:
            query_vec = query_vec / query_norm
        return np.dot(query_vec, embeddings_text.T)