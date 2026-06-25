"""
Chart Routes - Provides data for charts.
Contains only one public function: init_chart_routes().
"""

from flask import Blueprint, jsonify
import json
import traceback
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

chart_bp = Blueprint('charts', __name__, url_prefix='/api/chart')

def init_chart_routes(data_dir: Path, root_dir: Path):
    """Initialize chart routes with data directory."""
    
    @chart_bp.route('/<chart_type>', methods=['GET'])
    def get_chart_data(chart_type):
        """Return formatted data for the requested chart."""
        chart_type = chart_type.lower()
        results_dir = data_dir / 'results'
        
        if not results_dir.exists():
            results_dir = root_dir / 'data' / 'results'
            if not results_dir.exists():
                results_dir = root_dir
        
        try:
            if chart_type == 'model_comparison':
                return _get_model_comparison(results_dir)
            elif chart_type == 'bm25_params':
                return _get_bm25_params(results_dir)
            elif chart_type == 'hybrid_weights':
                return _get_hybrid_weights(results_dir)
            elif chart_type == 'serial_permutations':
                return _get_serial_permutations(results_dir)
            else:
                return jsonify({'error': f'Unsupported chart type: {chart_type}'}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    def _load_json_file(filename, results_dir):
        """Load a JSON file from various possible locations."""
        paths = [
            results_dir / filename,
            root_dir / 'data' / 'results' / filename,
            root_dir / filename,
            data_dir / 'results' / filename,
            data_dir / filename
        ]
        for p in paths:
            if p.exists():
                with open(p, 'r') as f:
                    return json.load(f)
        return None
    
    def _get_model_comparison(results_dir):
        """Data for model comparison chart."""
        data = _load_json_file('comparison_results.json', results_dir)
        if not data:
            return jsonify({'error': 'comparison_results.json not found'}), 404
        
        stats = data.get('stats', {})
        models = list(stats.keys())
        return jsonify({
            'models': models,
            'precision': [stats[m].get('avg_precision', 0) for m in models],
            'map': [stats[m].get('avg_map', 0) for m in models],
            'ndcg': [stats[m].get('avg_ndcg', 0) for m in models],
            'time': [stats[m].get('avg_time', 0) for m in models],
        })
    
    def _get_bm25_params(results_dir):
        """Data for BM25 parameter experiments."""
        data = _load_json_file('bm25_params_results.json', results_dir)
        if not data:
            return jsonify({'error': 'bm25_params_results.json not found'}), 404
        return jsonify(data)
    
    def _get_hybrid_weights(results_dir):
        """Data for hybrid weight experiments."""
        data = _load_json_file('hybrid_full_results.json', results_dir)
        if not data:
            return jsonify({'error': 'hybrid_full_results.json not found'}), 404
        
        results = data.get('results', {})
        weights_list = []
        for key, val in results.items():
            weights = val.get('weights', {})
            avg = val.get('average', {})
            weights_list.append({
                'tfidf': weights.get('tfidf', 0),
                'bert': weights.get('bert', 0),
                'bm25': weights.get('bm25', 0),
                'map': avg.get('map', 0),
                'ndcg': avg.get('ndcg', 0),
                'precision': avg.get('precision_at_k', 0)
            })
        return jsonify(weights_list)
    
    def _get_serial_permutations(results_dir):
        """Data for serial permutation experiments."""
        data = _load_json_file('serial_permutations_results.json', results_dir)
        if not data:
            return jsonify({'error': 'serial_permutations_results.json not found'}), 404
        
        results = data.get('results', [])
        seq_map = {}
        for entry in results:
            seq_name = entry.get('sequence_name', 'unknown')
            metrics = entry.get('metrics', {})
            seq_map[seq_name] = {
                'map': metrics.get('map', 0),
                'ndcg': metrics.get('ndcg', 0),
                'precision': metrics.get('precision_at_k', 0)
            }
        
        output = [{'sequence_name': k, **v} for k, v in seq_map.items()]
        return jsonify(output)
    
    return chart_bp