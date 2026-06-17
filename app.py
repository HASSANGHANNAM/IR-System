from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json
import os
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder='static', template_folder='templates')

ROOT = Path(__file__).resolve().parent
MODEL_DIRS = [ROOT / 'models', ROOT / 'saved_models']


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_sparse(path: Path):
    return sparse.load_npz(path)


def resolve_model_path(model_name: str):
    for root_dir in MODEL_DIRS:
        candidate = root_dir / model_name
        if candidate.exists():
            return candidate
    return None


def load_models():
    assets = {}

    tfidf_path = resolve_model_path('tfidf')
    if tfidf_path is not None:
        try:
            assets['tfidf'] = {
                'vectorizer': joblib.load(tfidf_path / 'tfidf_vectorizer.joblib'),
                'doc_vectors': load_sparse(tfidf_path / 'tfidf_vectors.npz'),
                'doc_ids': load_json(tfidf_path / 'doc_ids.json')
            }
        except Exception:
            pass

    bm25_path = resolve_model_path('bm25')
    if bm25_path is not None:
        try:
            assets['bm25'] = {
                'model': joblib.load(bm25_path / 'bm25_model.joblib'),
                'doc_ids': load_json(bm25_path / 'doc_ids.json')
            }
        except Exception:
            pass

    bert_path = resolve_model_path('bert')
    if bert_path is not None:
        try:
            assets['bert'] = {
                'model': joblib.load(bert_path / 'bert_model.joblib'),
                'doc_embeddings': np.load(bert_path / 'bert_embeddings.npy'),
                'doc_ids': load_json(bert_path / 'doc_ids.json')
            }
        except Exception:
            pass

    docs_path = ROOT / 'documents.json'
    if docs_path.exists():
        try:
            documents = load_json(docs_path)
            if isinstance(documents, dict):
                assets['documents'] = documents
            elif isinstance(documents, list):
                assets['documents'] = {str(item.get('doc_id')): item.get('text', '') for item in documents if isinstance(item, dict)}
        except Exception:
            pass

    return assets


def search_tfidf(query: str, assets: dict, top_k: int):
    query_vec = assets['vectorizer'].transform([query])
    sims = cosine_similarity(query_vec, assets['doc_vectors']).flatten()
    ranked = sorted(zip(assets['doc_ids'], sims), key=lambda x: x[1], reverse=True)
    return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in ranked[:top_k] if score > 0]


def search_bm25(query: str, assets: dict, top_k: int):
    tokenized = query.split()
    scores = assets['model'].get_scores(tokenized)
    ranked = sorted(zip(assets['doc_ids'], scores), key=lambda x: x[1], reverse=True)
    return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in ranked[:top_k] if score > 0]


def search_bert(query: str, assets: dict, top_k: int):
    query_vec = assets['model'].encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores = np.dot(assets['doc_embeddings'], query_vec.flatten())
    ranked = sorted(zip(assets['doc_ids'], scores), key=lambda x: x[1], reverse=True)
    return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in ranked[:top_k] if score > 0]


def search_hybrid(query: str, assets: dict, top_k: int, weights: list):
    score_maps = []
    if 'tfidf' in assets:
        tfidf_res = search_tfidf(query, assets['tfidf'], len(assets['tfidf']['doc_ids']))
        score_maps.append({item['doc_id']: item['score'] for item in tfidf_res})
    if 'bert' in assets:
        bert_res = search_bert(query, assets['bert'], len(assets['bert']['doc_ids']))
        score_maps.append({item['doc_id']: item['score'] for item in bert_res})
    if 'bm25' in assets:
        bm25_res = search_bm25(query, assets['bm25'], len(assets['bm25']['doc_ids']))
        score_maps.append({item['doc_id']: item['score'] for item in bm25_res})

    if not score_maps:
        return []

    total_weight = sum(weights)
    if total_weight == 0:
        weights = [1 / len(score_maps)] * len(score_maps)
    else:
        weights = [w / total_weight for w in weights]

    combined = {}
    for score_map, weight in zip(score_maps, weights):
        for doc_id, score in score_map.items():
            combined[doc_id] = combined.get(doc_id, 0.0) + score * weight

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in ranked[:top_k] if score > 0]


def refine_query_text(query: str):
    tokens = [token.lower() for token in query.split() if token.strip()]
    expanded = tokens + [f"{token}_enhanced" for token in tokens[:3]]
    normalized = ' '.join(tokens)
    return {'normalized': normalized, 'expanded': ' '.join(expanded)}


def get_document_text(doc_id: str, assets: dict):
    docs = assets.get('documents', {})
    return docs.get(doc_id, 'لا تتوفر نصوص للمستند. ضع ملف documents.json في جذر المشروع.')


def calculate_average_precision(retrieved: list, relevant: set):
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            score += hits / idx
    return score / hits if hits else 0.0


def calculate_map(retrieved_set: dict, relevants: dict):
    if not relevants:
        return 0.0
    ap_scores = []
    for qid, retrieved in retrieved_set.items():
        ap_scores.append(calculate_average_precision(retrieved, set(relevants.get(qid, []))))
    return float(np.mean(ap_scores) * 100) if ap_scores else 0.0


def calculate_mrr(retrieved_set: dict, relevants: dict):
    ranks = []
    for qid, retrieved in retrieved_set.items():
        relevant = set(relevants.get(qid, []))
        found = 0.0
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                found = 1.0 / idx
                break
        ranks.append(found)
    return float(np.mean(ranks) * 100) if ranks else 0.0

@app.route('/')
def home():
    assets = load_models()
    return render_template('index.html', models=list(assets.keys()))

@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.get_json() or {}
    query = data.get('query', '')
    model = data.get('model', 'tfidf')
    top_k = int(data.get('top_k', 10))
    weights = data.get('weights', [0.33, 0.33, 0.34])
    assets = load_models()

    if not query.strip():
        return jsonify({'error': 'Query is required.'}), 400

    if model == 'tfidf' and 'tfidf' in assets:
        results = search_tfidf(query, assets['tfidf'], top_k)
    elif model == 'bm25' and 'bm25' in assets:
        results = search_bm25(query, assets['bm25'], top_k)
    elif model == 'bert' and 'bert' in assets:
        results = search_bert(query, assets['bert'], top_k)
    elif model == 'hybrid':
        results = search_hybrid(query, assets, top_k, weights)
    else:
        return jsonify({'error': f'Model {model} not available.'}), 400

    for item in results:
        item['text'] = get_document_text(item['doc_id'], assets)

    return jsonify({'results': results, 'model': model, 'query': query})

@app.route('/api/refine', methods=['POST'])
def api_refine():
    data = request.get_json() or {}
    query = data.get('query', '')
    if not query.strip():
        return jsonify({'error': 'Query is required.'}), 400
    return jsonify(refine_query_text(query))

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.get_json() or {}
    retrieved = data.get('retrieved', [])
    relevant = data.get('relevant', [])
    top_k = int(data.get('top_k', 10))
    if not isinstance(retrieved, list) or not isinstance(relevant, list):
        return jsonify({'error': 'Sent lists are required.'}), 400
    retrieved_list = [str(item) for item in retrieved]
    relevant_set = set(str(item) for item in relevant)
    precision = sum(1 for i, doc in enumerate(retrieved_list[:top_k]) if doc in relevant_set) / max(1, top_k)
    map_score = calculate_map({'q1': retrieved_list}, {'q1': relevant_set})
    mrr_score = calculate_mrr({'q1': retrieved_list}, {'q1': relevant_set})
    return jsonify({'precision': precision * 100, 'map': map_score, 'mrr': mrr_score})

if __name__ == '__main__':
    app.run(debug=True)
