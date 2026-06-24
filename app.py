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
from sentence_transformers import SentenceTransformer
from scipy import sparse
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from spellchecker import SpellChecker
import faiss  # 🔥 استيراد FAISS

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / 'UI'
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
DB_PATH = ROOT / 'ir_data.db'
DOCUMENTS_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'
QUERIES_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'queries_cleaned.jsonl'
QRELS_PATHS = [
    DATA_DIR / 'test.qrels',
    DATA_DIR / 'beir' / 'webis-touche2020' / 'test.qrels',
    ROOT / 'beir' / 'webis-touche2020' / 'test.qrels',
]

app = Flask(__name__, static_folder=str(UI_DIR), static_url_path='')

CACHE = {}
QUERY_CACHE = {}
bert_model = None
embeddings_title_norm = None
embeddings_text_norm = None
doc_ids_list = []
bert_loaded = False
bm25_model = None
bm25_doc_ids = []
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# ===== متغيرات FAISS =====
faiss_index_text = None
faiss_index_title = None
faiss_doc_ids = []

def identity(x):
    return x

def split_tokens(x):
    return x.split()
    
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_npz(path):
    return sparse.load_npz(path)

def load_faiss_indexes():
    """تحميل فهارس FAISS من مجلد models (إن وجدت)."""
    global faiss_index_text, faiss_index_title, faiss_doc_ids
    faiss_text_path = MODELS_DIR / 'faiss_index_text.flat'
    faiss_title_path = MODELS_DIR / 'faiss_index_title.flat'
    faiss_ids_path = MODELS_DIR / 'faiss_doc_ids.pkl'

    if faiss_text_path.exists():
        try:
            faiss_index_text = faiss.read_index(str(faiss_text_path))
            print(f'✅ تم تحميل فهرس FAISS للنص من {faiss_text_path}')
        except Exception as e:
            print(f'⚠️ فشل تحميل فهرس FAISS للنص: {e}')
    else:
        print('ℹ️ فهرس FAISS للنص غير موجود، سيتم استخدام البحث العادي (القوة الغاشمة)')

    if faiss_title_path.exists():
        try:
            faiss_index_title = faiss.read_index(str(faiss_title_path))
            print(f'✅ تم تحميل فهرس FAISS للعنوان من {faiss_title_path}')
        except Exception as e:
            print(f'⚠️ فشل تحميل فهرس FAISS للعنوان: {e}')
    else:
        print('ℹ️ فهرس FAISS للعنوان غير موجود، سيتم استخدام البحث العادي')

    if faiss_ids_path.exists():
        try:
            with open(faiss_ids_path, 'rb') as f:
                faiss_doc_ids = pickle.load(f)
            print(f'✅ تم تحميل معرفات FAISS ({len(faiss_doc_ids)} وثيقة)')
        except Exception as e:
            print(f'⚠️ فشل تحميل معرفات FAISS: {e}')
    else:
        print('ℹ️ معرفات FAISS غير موجودة، سيتم استخدام doc_ids_list العادي')

def load_bert_resources():
    global bert_model
    global embeddings_title_norm
    global embeddings_text_norm
    global doc_ids_list
    global bert_loaded

    if bert_loaded and bert_model is not None and embeddings_text_norm is not None and doc_ids_list:
        return

    if bert_model is None:
        local_model_path = MODELS_DIR / 'bert_model'
        if local_model_path.exists():
            bert_model = SentenceTransformer(str(local_model_path))
            print('✅ تم تحميل BERT من المجلد المحلي')
        else:
            print('⚠️ النموذج المحلي غير موجود، يتم التحميل من Hugging Face...')
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            bert_model.save(str(local_model_path))
            print('✅ تم حفظ BERT في المجلد المحلي للمستقبل')

    if embeddings_text_norm is None:
        text_embeddings_path = MODELS_DIR / 'embeddings_text.npy'
        if not text_embeddings_path.exists():
            raise FileNotFoundError('embeddings_text.npy not found in models/')
        embeddings_text = np.load(text_embeddings_path, mmap_mode='r')
        norms_text = np.linalg.norm(embeddings_text, axis=1, keepdims=True)
        norms_text = np.where(norms_text == 0, 1.0, norms_text)
        embeddings_text_norm = embeddings_text / norms_text

    title_embeddings_path = MODELS_DIR / 'embeddings_title.npy'
    if title_embeddings_path.exists() and embeddings_title_norm is None:
        embeddings_title = np.load(title_embeddings_path, mmap_mode='r')
        norms_title = np.linalg.norm(embeddings_title, axis=1, keepdims=True)
        norms_title = np.where(norms_title == 0, 1.0, norms_title)
        embeddings_title_norm = embeddings_title / norms_title

    if not doc_ids_list:
        doc_ids_list = load_doc_ids()

    bert_loaded = True
    print('✅ تم تحميل BERT مع تطبيع المتجهات (تم التحميل عند أول طلب)')

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

def init_sqlite_qrels():
    qrels_path = next((path for path in QRELS_PATHS if path.exists()), None)
    if qrels_path is None:
        print('⚠️ qrels file not found in expected locations')
        return

    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            'CREATE TABLE IF NOT EXISTS qrels (query_id TEXT, doc_id TEXT, score REAL, PRIMARY KEY (query_id, doc_id))'
        )
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_qrels_query_id ON qrels(query_id)')
        cursor.execute('SELECT COUNT(*) FROM qrels')
        existing_count = cursor.fetchone()[0]

        if existing_count == 0:
            batch = []
            inserted = 0
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('query-id'):
                        continue
                    parts = line.split('\t')
                    if len(parts) < 3:
                        parts = line.split()
                    if len(parts) < 3:
                        continue
                    query_id, doc_id, score = parts[0], parts[1], parts[2]
                    try:
                        score_value = float(score)
                    except ValueError:
                        continue
                    batch.append((str(query_id), str(doc_id), score_value))
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO qrels (query_id, doc_id, score) VALUES (?, ?, ?)',
                            batch,
                        )
                        connection.commit()
                        inserted += len(batch)
                        batch.clear()
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO qrels (query_id, doc_id, score) VALUES (?, ?, ?)',
                    batch,
                )
                connection.commit()
                inserted += len(batch)
            print(f'✅ تم تخزين {inserted} صف qrels في SQLite من {qrels_path}')
        else:
            print(f'✅ SQLite qrels جاهز ويحتوي على {existing_count} صف')
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

# ============================================================
# 🔥 تحسين الاستعلام: التصحيح الإملائي فقط
# ============================================================
spell = SpellChecker()

def refine_query_text(query):
    """
    تحسين الاستعلام: تصحيح إملائي للكلمات الخاطئة فقط (بدون إضافة مرادفات).
    """
    if not query or not query.strip():
        return {'normalized': query, 'expanded': query}
    
    # 1. تجزئة الكلمات
    tokens = query.lower().split()
    
    # 2. تصحيح إملائي (Spell Checking)
    corrected_tokens = []
    misspelled = spell.unknown(tokens)
    for token in tokens:
        if token in misspelled:
            corrected = spell.correction(token)
            if corrected is not None and corrected != token:
                corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)
    
    # 3. إزالة التكرار مع الحفاظ على الترتيب (اختياري)
    seen = set()
    final_tokens = []
    for token in corrected_tokens:
        if token not in seen:
            seen.add(token)
            final_tokens.append(token)
    
    print(f"🔍 Original: {query}")
    print(f"✨ Corrected: {' '.join(final_tokens)}")
    
    return {
        'normalized': ' '.join(corrected_tokens),
        'expanded': ' '.join(final_tokens)
    }

def clean_query(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [STEMMER.stem(token) for token in tokens]
    return ' '.join(tokens)

# ============================
# دوال الحصول على درجات جميع الوثائق (للـ Hybrid)
# ============================

def get_all_tfidf_scores(query, assets):
    vectorizer = assets['vectorizer']
    cleaned_query = clean_query(query)
    query_vec = vectorizer.transform([cleaned_query])
    scores = query_vec.dot(assets['text_matrix_T']).toarray().flatten()
    return scores

def get_all_bm25_scores(query):
    global bm25_model
    if bm25_model is None:
        load_bm25_model()
    if bm25_model is None:
        raise ValueError('BM25 model not loaded')
    cleaned_query = clean_query(query)
    tokenized_query = cleaned_query.split()
    scores = np.array(bm25_model.get_scores(tokenized_query))
    return scores

def get_all_bert_scores(query):
    """تحسب درجات BERT لجميع الوثائق (القوة الغاشمة) - لا تستخدم FAISS."""
    global bert_model
    global embeddings_text_norm
    if bert_model is None or embeddings_text_norm is None:
        load_bert_resources()
    cleaned_query = clean_query(query)
    query_vec = bert_model.encode([cleaned_query])[0]
    query_norm = np.linalg.norm(query_vec)
    if query_norm != 0:
        query_vec = query_vec / query_norm
    
    scores = np.dot(query_vec, embeddings_text_norm.T)
    return scores

# ============================
# دالة البحث ب BERT مع دعم FAISS
# ============================

def search_bert(query, top_k=10, use_title=True, alpha=0.4, use_faiss=False):
    global bert_model
    global embeddings_title_norm
    global embeddings_text_norm
    global doc_ids_list
    global faiss_index_text, faiss_index_title, faiss_doc_ids

    if bert_model is None or embeddings_text_norm is None:
        load_bert_resources()

    cleaned_query = clean_query(query)
    query_vec = bert_model.encode([cleaned_query])[0]
    query_norm = np.linalg.norm(query_vec)
    if query_norm != 0:
        query_vec = query_vec / query_norm

    # إذا كان FAISS مفعلاً والفهارس موجودة
    if use_faiss and faiss_index_text is not None:
        # تحويل الاستعلام إلى float32
        query_vec_faiss = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec_faiss)  # تطبيع للتوافق مع الفهرس

        # البحث باستخدام FAISS
        k_faiss = max(top_k * 2, 50)  # نجلب ضعف العدد المطلوب لضمان تنوع
        if use_title and faiss_index_title is not None:
            distances, indices = faiss_index_title.search(query_vec_faiss, k_faiss)
        else:
            distances, indices = faiss_index_text.search(query_vec_faiss, k_faiss)

        # تحويل النتائج إلى scores (FAISS يعيد مسافات، نحتاج إلى تحويلها)
        # نأخذ فقط top_k
        top_indices = indices[0][:top_k]
        # استرجاع الدرجات الفعلية (نحتاج إلى حساب التشابه الفعلي)
        # بما أننا استخدمنا IndexFlatIP مع تطبيع، فإن المسافات هي التشابه (أكبر = أفضل)
        scores = distances[0][:top_k]  # هذه هي التشابهات

        results = []
        for idx, score in zip(top_indices, scores):
            if idx < len(faiss_doc_ids):
                doc_id = faiss_doc_ids[idx]
            else:
                # إذا لم تكن معرفات FAISS متوفرة، نستخدم doc_ids_list
                doc_id = doc_ids_list[idx] if idx < len(doc_ids_list) else str(idx)
            results.append({
                'doc_id': str(doc_id),
                'score': float(score)
            })
        return results

    # === الطريقة التقليدية (القوة الغاشمة) ===
    if use_title and embeddings_title_norm is not None:
        title_scores = np.dot(query_vec, embeddings_title_norm.T)
        text_scores = np.dot(query_vec, embeddings_text_norm.T)
        scores = alpha * title_scores + (1.0 - alpha) * text_scores
    else:
        scores = np.dot(query_vec, embeddings_text_norm.T)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            'doc_id': doc_ids_list[idx] if idx < len(doc_ids_list) else str(idx),
            'score': float(scores[idx]),
        }
        for idx in top_indices
        if scores[idx] > 0
    ]

# ============================
# دالة البحث الهجين الأساسية (مع دعم FAISS في الوضع المتوازي)
# ============================

def search_hybrid(query, hybrid_mode, weights, top_k=10, preprocessing='stemming', use_faiss=False):
    assets = get_search_assets(preprocessing)
    all_doc_ids = assets['doc_ids']
    total_docs = len(all_doc_ids)

    if hybrid_mode == 'parallel':
        if use_faiss and (weights.get('bert', 0) > 0) and faiss_index_text is not None:
            # == وضع FAISS: نحصل على مرشحين من BERT ثم ندمج درجات النماذج الأخرى ==
            # جلب عدد أكبر من المرشحين من BERT
            candidates_k = min(top_k * 3, total_docs, 200)  # نحصل على 200 كحد أقصى
            bert_results = search_bert(query, top_k=candidates_k, use_title=True, alpha=0.4, use_faiss=True)
            candidate_doc_ids = [item['doc_id'] for item in bert_results]
            if not candidate_doc_ids:
                return []

            # حساب درجات TF-IDF و BM25 لهذه الوثائق فقط
            # نحتاج إلى مصفوفة فرعية من TF-IDF
            # الطريقة: نقوم بإنشاء قائمة scores لكل وثيقة في candidate_doc_ids
            tfidf_scores_all = get_all_tfidf_scores(query, assets)  # نحصل على scores لكل الوثائق (قد يكون مكلفاً ولكن نضطر له)
            bm25_scores_all = get_all_bm25_scores(query) if weights.get('bm25', 0) > 0 else None

            # إنشاء قاموس للنتائج
            results_dict = {}
            for doc_id in candidate_doc_ids:
                # نبحث عن index الوثيقة
                try:
                    idx = all_doc_ids.index(doc_id)
                except ValueError:
                    continue
                scores = {}
                if weights.get('tfidf', 0) > 0:
                    scores['tfidf'] = tfidf_scores_all[idx]
                if weights.get('bm25', 0) > 0 and bm25_scores_all is not None:
                    scores['bm25'] = bm25_scores_all[idx]
                if weights.get('bert', 0) > 0:
                    # نبحث عن درجة BERT من نتائج FAISS
                    bert_score = next((item['score'] for item in bert_results if item['doc_id'] == doc_id), 0.0)
                    scores['bert'] = bert_score
                # تجميع scores
                final_score = 0.0
                for name, w in weights.items():
                    if name in scores:
                        final_score += w * scores[name]
                results_dict[doc_id] = final_score

            # ترتيب النتائج
            sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [{'doc_id': doc_id, 'score': float(score)} for doc_id, score in sorted_results]

        else:
            # == الوضع العادي (بدون FAISS) ==
            scores_dict = {}
            if weights.get('tfidf', 0) > 0:
                scores_dict['tfidf'] = get_all_tfidf_scores(query, assets)
            if weights.get('bm25', 0) > 0:
                scores_dict['bm25'] = get_all_bm25_scores(query)
            if weights.get('bert', 0) > 0:
                scores_dict['bert'] = get_all_bert_scores(query)

            if not scores_dict:
                raise ValueError('At least one model must have weight > 0')

            def normalize(scores):
                min_s = np.min(scores)
                max_s = np.max(scores)
                if max_s - min_s == 0:
                    return np.zeros_like(scores)
                return (scores - min_s) / (max_s - min_s)

            norm_scores = {}
            for name, scores in scores_dict.items():
                norm_scores[name] = normalize(scores)

            final_scores = np.zeros(total_docs)
            for name, scores in norm_scores.items():
                final_scores += weights[name] * scores

            top_indices = np.argsort(final_scores)[::-1][:top_k]
            results = []
            for idx in top_indices:
                if idx < len(all_doc_ids):
                    results.append({
                        'doc_id': all_doc_ids[idx],
                        'score': float(final_scores[idx])
                    })
            return results

    elif hybrid_mode == 'serial':
        # الوضع التسلسلي: لا يستخدم FAISS حالياً (يمكن إضافته لاحقاً)
        tfidf_scores = get_all_tfidf_scores(query, assets)
        initial_k = min(100, total_docs)
        top_tfidf_indices = np.argsort(tfidf_scores)[::-1][:initial_k]

        if len(top_tfidf_indices) == 0:
            return []

        all_bm25_scores = get_all_bm25_scores(query)
        bm25_scores_subset = all_bm25_scores[top_tfidf_indices]
        reranked_order = np.argsort(bm25_scores_subset)[::-1][:top_k]

        final_indices = top_tfidf_indices[reranked_order]
        final_scores = bm25_scores_subset[reranked_order]

        results = []
        for idx, score in zip(final_indices, final_scores):
            if idx < len(all_doc_ids):
                results.append({
                    'doc_id': all_doc_ids[idx],
                    'score': float(score)
                })
        return results

    else:
        raise ValueError("Invalid hybrid_mode. Must be 'parallel' or 'serial'")

# ============================
# دوال BM25 الأساسية
# ============================

def load_bm25_model():
    global bm25_model, bm25_doc_ids
    model_path = MODELS_DIR / 'bm25_model.pkl'
    if not model_path.exists():
        print('⚠️ BM25 model not found in models/')
        return False
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        bm25_model = data['bm25']
        bm25_doc_ids = data['doc_ids']
    print(f'✅ تم تحميل نموذج BM25 ({len(bm25_doc_ids)} وثيقة)')
    return True

def search_bm25(query, top_k=10):
    global bm25_model, bm25_doc_ids
    if bm25_model is None:
        if not load_bm25_model():
            raise ValueError('BM25 model not loaded')
    cleaned_query = clean_query(query)
    tokenized_query = cleaned_query.split()
    scores = bm25_model.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if idx < len(bm25_doc_ids):
            results.append({
                'doc_id': bm25_doc_ids[idx],
                'score': float(scores[idx])
            })
    return results

# ============================
# دوال البحث العادية (TF-IDF)
# ============================

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

# ============================
# دوال مساعدة للبيانات والتقييم
# ============================

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

def get_query_text_by_id(query_id):
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute('SELECT text FROM queries WHERE query_id = ?', (str(query_id),))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        connection.close()

def get_qrels_for_query(query_id):
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            'SELECT doc_id, score FROM qrels WHERE query_id = ? AND score > 0 ORDER BY score DESC, doc_id',
            (str(query_id),),
        )
        return [{'doc_id': str(doc_id), 'score': float(score)} for doc_id, score in cursor.fetchall()]
    finally:
        connection.close()

def get_document_titles(doc_ids):
    if not doc_ids:
        return {}
    placeholders = ','.join(['?'] * len(doc_ids))
    query = f'SELECT doc_id, title FROM documents WHERE doc_id IN ({placeholders})'
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(query, [str(doc_id) for doc_id in doc_ids])
        return {str(doc_id): title for doc_id, title in cursor.fetchall()}
    finally:
        connection.close()

def compute_evaluation_metrics(results, relevant_docs, top_k):
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

def build_evaluation_payload(results, query_id, top_k, preprocessing):
    relevant_docs = get_qrels_for_query(query_id)
    relevant_doc_ids = {item['doc_id'] for item in relevant_docs}
    title_map = get_document_titles([item['doc_id'] for item in relevant_docs] + [item['doc_id'] for item in results])

    for item in results:
        doc_id = str(item['doc_id'])
        item['relevant'] = doc_id in relevant_doc_ids
        item['relevance_score'] = next((rel['score'] for rel in relevant_docs if rel['doc_id'] == doc_id), 0.0)

    missing_relevant_docs = []
    result_doc_ids = {str(item['doc_id']) for item in results}
    for item in relevant_docs:
        if item['doc_id'] not in result_doc_ids:
            missing_relevant_docs.append(
                {
                    'doc_id': item['doc_id'],
                    'title': title_map.get(item['doc_id'], ''),
                    'score': item['score'],
                }
            )

    metrics = compute_evaluation_metrics(results, relevant_docs, top_k)
    return {
        'results': results,
        'query_id': str(query_id),
        'relevant_docs': relevant_docs,
        'missing_relevant_docs': missing_relevant_docs,
        'metrics': metrics,
        'preprocessing': preprocessing,
    }

# ============================
# Routes
# ============================

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        query_id = data.get('query_id', '')
        model = data.get('model', 'tfidf')
        top_k = int(data.get('top_k', 10))
        refine = bool(data.get('refine', False))
        preprocessing = data.get('preprocessing', 'stemming')
        evaluate = bool(data.get('evaluate', False))
        use_faiss = bool(data.get('use_faiss', False))  # 🔥 قراءة خيار FAISS

        hybrid_mode = data.get('hybrid_mode', 'parallel')
        weights = data.get('weights', None)

        bm25_k1 = data.get('bm25_k1', 2.0)
        bm25_b = data.get('bm25_b', 0.60)

        if not query.strip():
            return jsonify({'error': 'Query is required.'}), 400

        if refine:
            print(f"📝 Before refinement: {query}")
            query = refine_query_text(query)['expanded']
            print(f"📝 After refinement: {query}")
        else:
            print(f"📝 No refinement applied: {query}")

        if evaluate and not str(query_id).strip():
            return jsonify({'error': 'query_id is required when evaluation is enabled.'}), 400

        if model in ['bm25', 'hybrid']:
            if bm25_model is None:
                load_bm25_model()
            bm25_model.k1 = float(bm25_k1)
            bm25_model.b = float(bm25_b)
            print(f"🔧 BM25 params updated: k1={bm25_model.k1}, b={bm25_model.b}")

        cache_key = (query, str(query_id), model, top_k, preprocessing, evaluate, hybrid_mode, str(weights), bm25_k1, bm25_b, use_faiss)
        if cache_key in QUERY_CACHE:
            return jsonify(QUERY_CACHE[cache_key])

        results = []

        if model == 'tfidf':
            assets = get_search_assets(preprocessing)
            results = search_tfidf(query, assets, top_k)
            texts = get_documents_texts([item['doc_id'] for item in results])
            for item in results:
                item['text'] = texts.get(str(item['doc_id']), 'نص الوثيقة غير متوفر')

        elif model == 'bert':
            load_bert_resources()
            results = search_bert(query, top_k=top_k, use_title=True, alpha=0.4, use_faiss=use_faiss)
            texts = get_documents_texts([item['doc_id'] for item in results])
            for item in results:
                item['text'] = texts.get(str(item['doc_id']), 'نص الوثيقة غير متوفر')

        elif model == 'bm25':
            results = search_bm25(query, top_k)
            texts = get_documents_texts([item['doc_id'] for item in results])
            for item in results:
                item['text'] = texts.get(str(item['doc_id']), 'نص الوثيقة غير متوفر')

        elif model == 'hybrid':
            if hybrid_mode == 'parallel' and weights is None:
                return jsonify({'error': 'Weights are required for parallel hybrid mode.'}), 400

            results = search_hybrid(query, hybrid_mode, weights, top_k, preprocessing, use_faiss=use_faiss)
            texts = get_documents_texts([item['doc_id'] for item in results])
            for item in results:
                item['text'] = texts.get(str(item['doc_id']), 'نص الوثيقة غير متوفر')
        else:
            return jsonify({'error': f'Model {model} not available in this backend.'}), 400

        payload = {
            'results': results,
            'model': model,
            'query': query,
            'query_id': str(query_id),
            'preprocessing': preprocessing
        }

        if model == 'hybrid':
            payload['hybrid_mode'] = hybrid_mode
            if hybrid_mode == 'parallel':
                payload['weights'] = weights

        if evaluate:
            payload.update(build_evaluation_payload(results, query_id, top_k, preprocessing))

        QUERY_CACHE[cache_key] = payload
        return jsonify(payload)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/<doc_id>', methods=['GET'])
def get_document_by_id(doc_id):
    """
    جلب النص الكامل للوثيقة من قاعدة البيانات باستخدام doc_id.
    """
    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT doc_id, title, text FROM documents WHERE doc_id = ?", (str(doc_id),))
        row = cursor.fetchone()
        if row:
            return jsonify({
                'doc_id': row[0],
                'title': row[1],
                'text': row[2]
            })
        else:
            return jsonify({'error': 'Document not found'}), 404
    finally:
        connection.close()

@app.route('/api/queries', methods=['GET'])
def api_queries():
    try:
        return jsonify({'queries': get_queries_from_db()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
# ===== نقاط نهاية المخططات (Charts API) =====
# ============================================================

@app.route('/api/chart/<chart_type>', methods=['GET'])
def get_chart_data(chart_type):
    """
    إرجاع البيانات المنسقة لرسم المخطط المطلوب.
    """
    chart_type = chart_type.lower()
    results_dir = DATA_DIR / 'results'  # افترض أن ملفات JSON موجودة هنا
    if not results_dir.exists():
        results_dir = ROOT  # fallback إلى الجذر إذا لم يكن المجلد موجوداً

    try:
        if chart_type == 'model_comparison':
            file_path = results_dir / 'comparison_results.json'
            if not file_path.exists():
                # محاولة البحث في مسارات بديلة
                alt_paths = [ROOT / 'comparison_results.json', DATA_DIR / 'comparison_results.json']
                for p in alt_paths:
                    if p.exists():
                        file_path = p
                        break
                else:
                    return jsonify({'error': 'comparison_results.json not found'}), 404

            with open(file_path, 'r') as f:
                data = json.load(f)
            stats = data.get('stats', {})
            models = list(stats.keys())
            # استخراج المتوسطات
            precision = [stats[m].get('avg_precision', 0) for m in models]
            map_vals = [stats[m].get('avg_map', 0) for m in models]
            ndcg = [stats[m].get('avg_ndcg', 0) for m in models]
            time = [stats[m].get('avg_time', 0) for m in models]
            return jsonify({
                'models': models,
                'precision': precision,
                'map': map_vals,
                'ndcg': ndcg,
                'time': time
            })

        elif chart_type == 'bm25_params':
            file_path = results_dir / 'bm25_params_results.json'
            if not file_path.exists():
                alt_paths = [ROOT / 'bm25_params_results.json', DATA_DIR / 'bm25_params_results.json']
                for p in alt_paths:
                    if p.exists():
                        file_path = p
                        break
                else:
                    return jsonify({'error': 'bm25_params_results.json not found'}), 404

            with open(file_path, 'r') as f:
                data = json.load(f)
            # البيانات عبارة عن قائمة من الكائنات
            return jsonify(data)  # نعيدها كما هي، ستتم معالجتها في الواجهة

        elif chart_type == 'hybrid_weights':
            file_path = results_dir / 'hybrid_full_results.json'
            if not file_path.exists():
                alt_paths = [ROOT / 'hybrid_full_results.json', DATA_DIR / 'hybrid_full_results.json']
                for p in alt_paths:
                    if p.exists():
                        file_path = p
                        break
                else:
                    return jsonify({'error': 'hybrid_full_results.json not found'}), 404

            with open(file_path, 'r') as f:
                data = json.load(f)
            # استخراج جميع تركيبات الأوزان مع متوسطاتها
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

        elif chart_type == 'serial_permutations':
            file_path = results_dir / 'serial_permutations_results.json'
            if not file_path.exists():
                alt_paths = [ROOT / 'serial_permutations_results.json', DATA_DIR / 'serial_permutations_results.json']
                for p in alt_paths:
                    if p.exists():
                        file_path = p
                        break
                else:
                    return jsonify({'error': 'serial_permutations_results.json not found'}), 404

            with open(file_path, 'r') as f:
                data = json.load(f)
            # استخراج النتائج لكل تسلسل
            results = data.get('results', [])
            # تجميع حسب التسلسل وحساب المتوسط
            seq_map = {}
            for entry in results:
                seq_name = entry.get('sequence_name', 'unknown')
                metrics = entry.get('metrics', {})
                seq_map[seq_name] = {
                    'map': metrics.get('map', 0),
                    'ndcg': metrics.get('ndcg', 0),
                    'precision': metrics.get('precision_at_k', 0)
                }
            # تحويل إلى قائمة
            output = [{'sequence_name': k, **v} for k, v in seq_map.items()]
            return jsonify(output)

        else:
            return jsonify({'error': f'Unsupported chart type: {chart_type}'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_sqlite_documents()
    init_sqlite_queries()
    init_sqlite_qrels()
    load_faiss_indexes()  # 🔥 تحميل فهارس FAISS عند بدء التشغيل
    print('🚀 Server starting... (BERT, BM25, and Hybrid will load on first use)')
    app.run(debug=False, host='0.0.0.0', port=5000)