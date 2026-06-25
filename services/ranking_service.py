"""
Ranking Service - Handles evaluation and re-ranking.
Contains one class: RankingService.
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RankingService:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager
    
    def compute_evaluation_metrics(self, results: List[Dict], relevant_docs: List[Dict], top_k: int) -> Dict:
        top_results = results[:top_k]
        relevant_scores = {str(item['doc_id']): float(item['score']) for item in relevant_docs}
        relevant_set = set(relevant_scores)
        
        hits = 0
        precision_sum = 0.0
        dcg = 0.0
        
        for index, item in enumerate(top_results, start=1):
            doc_id = str(item['doc_id'])
            if doc_id in relevant_set:
                hits += 1
                precision_sum += hits / index
                relevance = relevant_scores.get(doc_id, 0.0)
                dcg += (2 ** relevance - 1) / np.log2(index + 1)
        
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:top_k]
        idcg = sum((2 ** score - 1) / np.log2(index + 2) for index, score in enumerate(ideal_scores))
        
        precision_at_k = hits / top_k if top_k else 0.0
        map_score = precision_sum / len(relevant_set) if relevant_set else 0.0
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall_at_k = hits / len(relevant_set) if relevant_set else 0.0
        
        return {
            'precision_at_k': round(float(precision_at_k), 6),
            'map': round(float(map_score), 6),
            'ndcg': round(float(ndcg), 6),
            'recall_at_k': round(float(recall_at_k), 6),
        }
    
    def build_evaluation_payload(self, results: List[Dict], query_id: str, top_k: int) -> Dict:
        relevant_docs = self.db.get_qrels_for_query(query_id)
        relevant_doc_ids = {item['doc_id'] for item in relevant_docs}
        
        all_doc_ids = [item['doc_id'] for item in relevant_docs] + [item['doc_id'] for item in results]
        title_map = self.db.get_document_titles(all_doc_ids)
        
        for item in results:
            doc_id = str(item['doc_id'])
            item['relevant'] = doc_id in relevant_doc_ids
            item['relevance_score'] = next((rel['score'] for rel in relevant_docs if rel['doc_id'] == doc_id), 0.0)
        
        result_doc_ids = {str(item['doc_id']) for item in results}
        missing_relevant_docs = []
        for item in relevant_docs:
            if item['doc_id'] not in result_doc_ids:
                missing_relevant_docs.append({
                    'doc_id': item['doc_id'],
                    'title': title_map.get(item['doc_id'], ''),
                    'score': item['score'],
                })
        
        metrics = self.compute_evaluation_metrics(results, relevant_docs, top_k)
        
        return {
            'results': results,
            'query_id': str(query_id),
            'relevant_docs': relevant_docs,
            'missing_relevant_docs': missing_relevant_docs,
            'metrics': metrics,
        }