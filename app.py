from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
import json
import pickle
import sqlite3
import string
import traceback
import unicodedata
from functools import lru_cache

import numpy as np
from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / 'UI'
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
DB_PATH = ROOT / 'ir_data.db'
DOCUMENTS_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
QUERIES_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'queries_cleaned.jsonl'

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path='')

CACHE = {}
QUERY_CACHE = {}
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()


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
    if DOCUMENTS_JSONL_PATH.exists():
        doc_ids = []
        with open(DOCUMENTS_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                doc_id = data.get('doc_id')
                if doc_id:
                    doc_ids.append(str(doc_id))
        if doc_ids:
            print(f"✅ تم تحميل {len(doc_ids)} معرف وثيقة من corpus_original.jsonl")
            return doc_ids

    for candidate in [MODELS_DIR / 'doc_ids.json', MODELS_DIR / 'doc_ids.pkl', MODELS_DIR / 'doc_ids.txt']:
        if candidate.exists():
            if candidate.suffix == '.json':
                with open(candidate, 'r', encoding='utf-8') as f:
                    return json.load(f)
            if candidate.suffix == '.pkl':
                return load_pickle(candidate)
            with open(candidate, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]

    matrix = load_text_matrix()
    if matrix is not None:
        print('⚠️ تحذير: لم يتم العثور على معرفات حقيقية، سيتم استخدام أرقام الصفوف.')
        return [str(i) for i in range(matrix.shape[0])]
    return []


def init_sqlite_documents():
    if not DOCUMENTS_JSONL_PATH.exists():
        print(f'⚠️ corpus file not found: {DOCUMENTS_JSONL_PATH}')
        return

    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, title TEXT, text TEXT)')
        cursor.execute('SELECT COUNT(*) FROM documents')
        existing_count = cursor.fetchone()[0]

        if existing_count == 0:
            batch = []
            inserted = 0
            with open(DOCUMENTS_JSONL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    doc_id = str(data.get('doc_id', ''))
                    if not doc_id:
                        continue
                    title = data.get('title', '')
                    text = data.get('text', '')
                    batch.append((doc_id, title, text))
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO documents (doc_id, title, text) VALUES (?, ?, ?)',
                            batch,
                        )
                        connection.commit()
                        inserted += len(batch)
                        batch.clear()
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO documents (doc_id, title, text) VALUES (?, ?, ?)',
                    batch,
                )
                connection.commit()
                inserted += len(batch)
            print(f'✅ تم تخزين {inserted} وثيقة في SQLite')
        else:
            print(f'✅ SQLite جاهز ويحتوي على {existing_count} وثيقة')
    finally:
        connection.close()


def init_sqlite_queries():
    if not QUERIES_JSONL_PATH.exists():
        print(f'⚠️ queries file not found: {QUERIES_JSONL_PATH}')
        return

    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS queries (query_id TEXT PRIMARY KEY, text TEXT, description TEXT, narrative TEXT)'
        )
        cursor.execute('SELECT COUNT(*) FROM queries')
        existing_count = cursor.fetchone()[0]

        if existing_count == 0:
            batch = []
            inserted = 0
            with open(QUERIES_JSONL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    query_id = str(data.get('query_id', ''))
                    if not query_id:
                        continue
                    text = data.get('text') or data.get('original_text') or data.get('query') or ''
                    description = data.get('description', '')
                    narrative = data.get('narrative', '')
                    batch.append((query_id, text, description, narrative))
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO queries (query_id, text, description, narrative) VALUES (?, ?, ?, ?)',
                            batch,
                        )
                        connection.commit()
                        inserted += len(batch)
                        batch.clear()
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO queries (query_id, text, description, narrative) VALUES (?, ?, ?, ?)',
                    batch,
                )
                connection.commit()
                inserted += len(batch)
            print(f'✅ تم تخزين {inserted} كويري في SQLite')
        else:
            print(f'✅ SQLite queries جاهز ويحتوي على {existing_count} كويري')
    finally:
        connection.close()


def get_search_assets(preprocessing):
    key = preprocessing.lower() if preprocessing else 'stemming'
    if key in CACHE:
        return CACHE[key]

    text_matrix = load_text_matrix()
    if text_matrix is None:
        raise FileNotFoundError('tfidf_text_matrix.npz not found in models/')

    vectorizer = load_vectorizer()
    if vectorizer is None:
        raise FileNotFoundError('tfidf_vectorizer.pkl not found in models/')

    assets = {
        'text_matrix': text_matrix,
        'text_matrix_T': text_matrix.T,
        'vectorizer': vectorizer,
        'doc_ids': load_doc_ids(),
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
    scores = query_vec.dot(assets['text_matrix_T']).toarray().flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    doc_ids = assets['doc_ids']
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            doc_id = doc_ids[idx] if idx < len(doc_ids) else str(idx)
            results.append({'doc_id': doc_id, 'score': float(score)})
    return results


def get_documents_texts(doc_ids):
    if not doc_ids:
        return {}

    placeholders = ','.join(['?'] * len(doc_ids))
    query = f'SELECT doc_id, text FROM documents WHERE doc_id IN ({placeholders})'
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(query, [str(doc_id) for doc_id in doc_ids])
        return {str(doc_id): text for doc_id, text in cursor.fetchall()}
    finally:
        connection.close()


def get_queries_from_db():
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        cursor.execute('SELECT query_id, text, description, narrative FROM queries ORDER BY CAST(query_id AS INTEGER), query_id')
        rows = cursor.fetchall()
        return [
            {
                'query_id': row['query_id'],
                'text': row['text'],
                'description': row['description'],
                'narrative': row['narrative'],
            }
            for row in rows
        ]
    finally:
        connection.close()


@lru_cache(maxsize=128)
def get_document_text_cached(doc_id):
    jsonl_path = DOCUMENTS_JSONL_PATH
    if not jsonl_path.exists():
        return 'نص الوثيقة غير متوفر (الملف غير موجود)'
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if str(data.get('doc_id', '')) == str(doc_id):
                    return data.get('text', '')
    except Exception:
        pass
    return 'نص الوثيقة غير متوفر'


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

        cache_key = (query, model, top_k, preprocessing)
        if cache_key in QUERY_CACHE:
            return jsonify(QUERY_CACHE[cache_key])

        assets = get_search_assets(preprocessing)

        if model != 'tfidf':
            return jsonify({'error': f'Model {model} not available in this backend. Use tfidf.'}), 400

        results = search_tfidf(query, assets, top_k)
        doc_id_list = [item['doc_id'] for item in results]
        texts = get_documents_texts(doc_id_list)
        for item in results:
            item['text'] = texts.get(str(item['doc_id']), 'نص الوثيقة غير متوفر')

        payload = {'results': results, 'model': model, 'query': query, 'preprocessing': preprocessing}
        QUERY_CACHE[cache_key] = payload
        return jsonify(payload)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/queries', methods=['GET'])
def api_queries():
    try:
        return jsonify({'queries': get_queries_from_db()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    init_sqlite_documents()
    init_sqlite_queries()
    print('Server starting...')
    app.run(debug=False, host='0.0.0.0', port=5000)