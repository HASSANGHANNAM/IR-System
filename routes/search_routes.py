"""
Search Routes - Registers the search Blueprint.
Contains only one public function: init_search_routes().
"""

from flask import Blueprint, request, jsonify
import traceback
import logging

from utils.text_utils import refine_query_text

logger = logging.getLogger(__name__)

search_bp = Blueprint('search', __name__, url_prefix='/api')

def init_search_routes(retrieval_service, ranking_service, db_manager, cache_manager):
    """Initialize search routes with dependency injection."""
    
    SEARCH_HANDLERS = {
        'tfidf': retrieval_service.search_tfidf,
        'bm25': retrieval_service.search_bm25,
        'bert': retrieval_service.search_bert,
        'hybrid': retrieval_service.search_hybrid,
    }
    
    @search_bp.route('/search', methods=['POST'])
    def api_search():
        """Main search endpoint."""
        try:
            data = request.get_json() or {}
            query = data.get('query', '')
            query_id = data.get('query_id', '')
            model = data.get('model', 'tfidf')
            top_k = int(data.get('top_k', 10))
            refine = bool(data.get('refine', False))
            evaluate = bool(data.get('evaluate', False))
            use_faiss = bool(data.get('use_faiss', False))
            hybrid_mode = data.get('hybrid_mode', 'parallel')
            weights = data.get('weights', None)
            bm25_k1 = data.get('bm25_k1', 2.0)
            bm25_b = data.get('bm25_b', 0.60)
            
            if not query.strip():
                return jsonify({'error': 'Query is required.'}), 400
            
            # ===== التحقق من query_id للتقييم =====
            if evaluate:
                if not query_id:
                    return jsonify({
                        'error': 'query_id is required when evaluate is enabled.'
                    }), 400
                
                query_text = db_manager.get_query_text_by_id(query_id)
                if query_text is None:
                    return jsonify({
                        'warning': f'Query ID "{query_id}" not found in database.',
                        'message': 'Evaluation is only available for the 49 predefined queries.',
                        'evaluation_available': False,
                        'suggestion': 'Use one of the available query IDs from /api/queries'
                    }), 200  # 200 عشان الواجهة تتعامل معاه كـ response عادي مش error
            
            # Apply query refinement if requested
            if refine:
                logger.info(f"Before refinement: {query}")
                query = refine_query_text(query)['expanded']
                logger.info(f"After refinement: {query}")
            
            # ===== التحقق من الكاش قبل التنفيذ =====
            weights_key = None
            if weights:
                weights_key = tuple(sorted(weights.items()))
            
            cache_key = (
                query,
                str(query_id),
                model,
                top_k,
                refine,
                evaluate,
                use_faiss,
                hybrid_mode,
                weights_key,
                bm25_k1,
                bm25_b
            )
            
            cached_result = cache_manager.get(cache_key, 'query')
            if cached_result is not None:
                logger.info(f"Cache hit for query: {query[:30]}...")
                return jsonify(cached_result)
            
            logger.info(f"Cache miss for query: {query[:30]}...")
            
            # ===== تنفيذ البحث =====
            handler = SEARCH_HANDLERS.get(model)
            if handler is None:
                return jsonify({'error': f'Model {model} not available'}), 400
            
            # Execute search based on model type
            if model == 'bm25':
                results = handler(query, top_k, k1=bm25_k1, b=bm25_b)
            elif model == 'hybrid':
                if hybrid_mode == 'parallel' and weights is None:
                    return jsonify({'error': 'Weights are required for parallel hybrid mode.'}), 400
                results = handler(query, hybrid_mode, weights, top_k, use_faiss)
            elif model == 'bert':
                results = handler(query, top_k, use_title=True, alpha=0.4, use_faiss=use_faiss)
            else:
                results = handler(query, top_k)
            
            # Fetch full texts for the results
            doc_ids = [item['doc_id'] for item in results]
            texts = db_manager.get_documents_texts(doc_ids)
            for item in results:
                item['text'] = texts.get(str(item['doc_id']), 'Document text not available')
            
            # Build response payload
            payload = {
                'results': results,
                'model': model,
                'query': query,
                'query_id': str(query_id),
            }
            
            if model == 'hybrid':
                payload['hybrid_mode'] = hybrid_mode
                if hybrid_mode == 'parallel':
                    payload['weights'] = weights
            
            # Add evaluation metrics if requested
            if evaluate and query_id:
                eval_payload = ranking_service.build_evaluation_payload(results, query_id, top_k)
                payload.update(eval_payload)
            
            # ===== تخزين النتيجة في الكاش =====
            cache_manager.set(cache_key, payload, 'query')
            
            return jsonify(payload)
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @search_bp.route('/document/<doc_id>', methods=['GET'])
    def get_document(doc_id):
        """Fetch a full document by its ID."""
        doc = db_manager.get_document_by_id(doc_id)
        if doc:
            return jsonify(doc)
        return jsonify({'error': 'Document not found'}), 404
    
    @search_bp.route('/queries', methods=['GET'])
    def get_queries():
        """Fetch all queries from the database."""
        try:
            return jsonify({'queries': db_manager.get_queries()})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return search_bp