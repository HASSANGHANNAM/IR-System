"""
Admin Routes - Registers the admin Blueprint.
Contains only one public function: init_admin_routes().
"""

from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

def init_admin_routes(model_manager, cache_manager):
    """Initialize admin routes with dependency injection."""
    
    @admin_bp.route('/models/load/<model_name>', methods=['POST'])
    def load_model(model_name):
        """Load a specific model."""
        try:
            if model_name == 'bm25':
                success = model_manager.load_bm25()
            elif model_name == 'faiss':
                success = model_manager.load_faiss_indexes()
            elif model_name == 'tfidf':
                success = model_manager.load_tfidf()
            elif model_name == 'bert':
                success = model_manager.load_bert()
            else:
                return jsonify({'error': f'Unknown model: {model_name}'}), 400
            
            if success:
                return jsonify({'status': 'loaded', 'model': model_name})
            else:
                return jsonify({'error': f'Failed to load {model_name}'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/models/unload', methods=['POST'])
    def unload_models():
        """Clear all models from memory."""
        try:
            model_manager.clear_all()
            return jsonify({'status': 'unloaded', 'message': 'All models cleared'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/cache/clear', methods=['POST'])
    def clear_cache():
        """Clear the cache."""
        try:
            cache_type = request.get_json().get('type', 'all')
            cache_manager.clear(cache_type if cache_type != 'all' else None)
            return jsonify({'status': 'cleared', 'cache_type': cache_type})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @admin_bp.route('/status', methods=['GET'])
    def get_status():
        """Get system status."""
        return jsonify({
            'models': {
                'bm25': model_manager.bm25_loaded,
                'faiss': model_manager.faiss_loaded,
                'tfidf': model_manager.tfidf_loaded,
                'bert': model_manager.bert_loaded,
            },
            'cache_size': len(cache_manager._query_cache),
            'documents_count': len(model_manager.doc_ids_list),
        })
    
    return admin_bp