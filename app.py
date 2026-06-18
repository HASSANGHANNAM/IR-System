from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import json
import pickle
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
import os
import string
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / 'UI'
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path='')

CACHE = {}
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def identity(x):
    return x

def split_tokens(x):
    return x.split()
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_npz(path):
    return sparse.load_npz(path)

def load_text_matrix():
    path = MODELS_DIR / 'tfidf_text_matrix.npz'
    if path.exists():
        return load_npz(path)
    return None

def load_vectorizer():
    path = MODELS_DIR / 'tfidf_vectorizer.pkl'
    if path.exists():
        return load_pickle(path)
    return None
def load_doc_ids():
    # 1. أولاً: حاول قراءة المعرفات من corpus_original.jsonl
    jsonl_path = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
    if jsonl_path.exists():
        doc_ids = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc_id = data.get('doc_id')
                if doc_id:
                    doc_ids.append(str(doc_id))
        if doc_ids:
            print(f"✅ تم تحميل {len(doc_ids)} معرف وثيقة من corpus_original.jsonl")
            return doc_ids
    
    # 2. ثانياً: حاول تحميل من ملفات doc_ids المختلفة (إن وجدت)
    for candidate in [MODELS_DIR / 'doc_ids.json', MODELS_DIR / 'doc_ids.pkl', MODELS_DIR / 'doc_ids.txt']:
        if candidate.exists():
            if candidate.suffix == '.json':
                with open(candidate, 'r') as f:
                    return json.load(f)
            elif candidate.suffix == '.pkl':
                return load_pickle(candidate)
            else:
                with open(candidate, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
    
    # 3. أخيراً: إذا لم نجد أي شيء، ننشئ قائمة افتراضية (تحذير)
    matrix = load_text_matrix()
    if matrix is not None:
        print("⚠️ تحذير: لم يتم العثور على معرفات حقيقية، سيتم استخدام أرقام الصفوف.")
        return [str(i) for i in range(matrix.shape[0])]
    return []
def get_search_assets(preprocessing):
    key = preprocessing.lower() if preprocessing else 'stemming'
    if key in CACHE:
        return CACHE[key]

    text_matrix = load_text_matrix()
    if text_matrix is None:
        raise FileNotFoundError("tfidf_text_matrix.npz not found in models/")

    vectorizer = load_vectorizer()
    if vectorizer is None:
        raise FileNotFoundError("tfidf_vectorizer.pkl not found in models/")

    doc_ids = load_doc_ids()

    assets = {
        'text_matrix': text_matrix,
        'vectorizer': vectorizer,
        'doc_ids': doc_ids,
    }
    CACHE[key] = assets
    return assets

def refine_query_text(query):
    tokens = [token.lower() for token in query.split() if token.strip()]
    expanded = tokens + [f'{token}_enhanced' for token in tokens[:3]]
    normalized = ' '.join(tokens)
    return {'normalized': normalized, 'expanded': ' '.join(expanded)}

def clean_query(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [STEMMER.stem(token) for token in tokens]
    return ' '.join(tokens)

def search_tfidf(query, assets, top_k):
    vectorizer = assets['vectorizer']
    cleaned_query = clean_query(query)
    query_vec = vectorizer.transform([cleaned_query])
    text_matrix = assets['text_matrix']
    scores = cosine_similarity(query_vec, text_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    doc_ids = assets['doc_ids']
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            doc_id = doc_ids[idx] if idx < len(doc_ids) else str(idx)
            results.append({'doc_id': doc_id, 'score': float(score)})
    return results

# 🔥 دالة لجلب النص من الملف عند الطلب (مع تخزين مؤقت بسيط)
@lru_cache(maxsize=128)
def get_document_text_cached(doc_id):
    jsonl_path = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
    if not jsonl_path.exists():
        return "نص الوثيقة غير متوفر (الملف غير موجود)"
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if str(data.get('doc_id', '')) == str(doc_id):
                    return data.get('text', '')
    except Exception:
        pass
    return "نص الوثيقة غير متوفر"

def get_document_text(doc_id):
    return get_document_text_cached(doc_id)

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        model = data.get('model', 'tfidf')
        top_k = int(data.get('top_k', 10))
        refine = bool(data.get('refine', False))
        preprocessing = data.get('preprocessing', 'stemming')

        if not query.strip():
            return jsonify({'error': 'Query is required.'}), 400

        if refine:
            query = refine_query_text(query)['expanded']

        assets = get_search_assets(preprocessing)

        if model != 'tfidf':
            return jsonify({'error': f'Model {model} not available in this backend. Use tfidf.'}), 400

        results = search_tfidf(query, assets, top_k)
        for item in results:
            item['text'] = get_document_text(item['doc_id'])

        return jsonify({'results': results, 'model': model, 'query': query, 'preprocessing': preprocessing})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server starting...")
    app.run(debug=False, host='0.0.0.0', port=5000)