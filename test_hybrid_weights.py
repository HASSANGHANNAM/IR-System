#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت مستقل لاختبار جميع الأوزان الهجين (Parallel Hybrid)
على جميع الاستعلامات، مع معالجة الدفعات (Batch Processing)
بدون الحاجة لتعديل ملف app.py الأصلي.
"""

# ============================================================
# 🔥 الخطوة الأهم: تعريف الدوال التي يحتاجها pickle لتحميل الـ vectorizer
#    (دون الحاجة لتعديل app.py)
# ============================================================
def identity(x):
    return x

def split_tokens(x):
    return x.split()

# ============================================================
# استيراد المكتبات والدوال من app.py
# ============================================================
import json
import sqlite3
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# استيراد الدوال المطلوبة من app.py (باستثناء Flask)
from app import (
    get_search_assets,
    get_all_tfidf_scores,
    get_all_bm25_scores,
    get_all_bert_scores,
    load_bm25_model,
    load_bert_resources,
    compute_evaluation_metrics,
    get_qrels_for_query,
    get_documents_texts,
    DB_PATH,
)

# ============================================================
# دوال مساعدة
# ============================================================

def get_all_queries():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT query_id, text FROM queries ORDER BY CAST(query_id AS INTEGER)")
        rows = cursor.fetchall()
        return [(str(qid), text) for qid, text in rows if text and text.strip()]
    finally:
        conn.close()

def generate_weight_combinations(step=0.2, include_zero=True):
    weights = np.arange(0.0, 1.0 + step, step).round(2).tolist()
    if not include_zero:
        weights = [w for w in weights if w > 0]
    combos = []
    for w_tfidf in weights:
        for w_bert in weights:
            for w_bm25 in weights:
                if abs(w_tfidf + w_bert + w_bm25 - 1.0) < 0.001:
                    if w_tfidf > 0 or w_bert > 0 or w_bm25 > 0:
                        combos.append({
                            'tfidf': round(w_tfidf, 2),
                            'bert': round(w_bert, 2),
                            'bm25': round(w_bm25, 2)
                        })
    return combos

# ============================================================
# الدالة الرئيسية (مع التخزين المؤقت للدرجات)
# ============================================================

def run_experiment_full(output_file='hybrid_full_results.json', top_k=10, step=0.2):
    print(f"🚀 بدء التجربة الشاملة...")
    print(f"   - عدد النتائج (Top-K): {top_k}")
    print(f"   - خطوة الأوزان: {step}")

    # تحميل النماذج
    print("⏳ تحميل النماذج...")
    load_bert_resources()
    load_bm25_model()
    assets = get_search_assets('stemming')
    print("✅ تم تحميل جميع النماذج.")

    queries = get_all_queries()
    print(f"✅ تم تحميل {len(queries)} استعلام.")

    weight_combos = generate_weight_combinations(step=step, include_zero=True)
    print(f"✅ تم توليد {len(weight_combos)} تركيبة أوزان.")

    # ----------------------------------------------------------
    # المرحلة 1: حساب درجات كل نموذج لكل استعلام (مرة واحدة)
    # ----------------------------------------------------------
    print("\n⏳ المرحلة 1: حساب درجات النماذج لكل استعلام...")
    scores_cache = {}

    for qid, query_text in tqdm(queries, desc="حساب الدرجات"):
        try:
            scores_tfidf = get_all_tfidf_scores(query_text, assets)
            scores_bm25 = get_all_bm25_scores(query_text)
            scores_bert = get_all_bert_scores(query_text)

            def normalize(s):
                mn, mx = np.min(s), np.max(s)
                return (s - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(s)

            scores_cache[qid] = {
                'tfidf_norm': normalize(scores_tfidf),
                'bm25_norm': normalize(scores_bm25),
                'bert_norm': normalize(scores_bert),
                'query_text': query_text
            }
        except Exception as e:
            print(f"   ⚠️ خطأ في الاستعلام {qid}: {e}")
            continue

    print(f"✅ تم حساب درجات {len(scores_cache)} استعلام.")

    # ----------------------------------------------------------
    # المرحلة 2: تطبيق الأوزان وحساب المقاييس لكل استعلام
    # ----------------------------------------------------------
    print("\n⏳ المرحلة 2: تطبيق الأوزان وحساب المقاييس...")

    final_results = {
        'top_k': top_k,
        'step': step,
        'total_queries': len(scores_cache),
        'total_weights': len(weight_combos),
        'results': {}
    }

    all_doc_ids = assets['doc_ids']

    for w_idx, weights in enumerate(tqdm(weight_combos, desc="تطبيق الأوزان")):
        weight_key = f"tfidf{weights['tfidf']:.2f}_bert{weights['bert']:.2f}_bm25{weights['bm25']:.2f}"
        final_results['results'][weight_key] = {
            'weights': weights,
            'queries': {}
        }

        all_metrics = []
        for qid, cached in scores_cache.items():
            try:
                final_scores = (weights['tfidf'] * cached['tfidf_norm'] +
                                weights['bert'] * cached['bert_norm'] +
                                weights['bm25'] * cached['bm25_norm'])

                top_indices = np.argsort(final_scores)[::-1][:top_k]
                results = []
                for idx in top_indices:
                    if idx < len(all_doc_ids):
                        results.append({
                            'doc_id': all_doc_ids[idx],
                            'score': float(final_scores[idx])
                        })

                texts = get_documents_texts([item['doc_id'] for item in results])
                for item in results:
                    item['text'] = texts.get(str(item['doc_id']), '')

                relevant_docs = get_qrels_for_query(qid)
                metrics = compute_evaluation_metrics(results, relevant_docs, top_k)

                final_results['results'][weight_key]['queries'][qid] = {
                    'query_text': cached['query_text'],
                    'metrics': metrics
                }
                all_metrics.append(metrics)

            except Exception as e:
                print(f"   ⚠️ خطأ في الاستعلام {qid} مع الوزن {weight_key}: {e}")
                final_results['results'][weight_key]['queries'][qid] = {
                    'query_text': cached.get('query_text', ''),
                    'error': str(e)
                }

        if all_metrics:
            avg_p = np.mean([m['precision_at_k'] for m in all_metrics])
            avg_map = np.mean([m['map'] for m in all_metrics])
            avg_ndcg = np.mean([m['ndcg'] for m in all_metrics])
            final_results['results'][weight_key]['average'] = {
                'precision_at_k': round(avg_p, 6),
                'map': round(avg_map, 6),
                'ndcg': round(avg_ndcg, 6)
            }

        # حفظ التقدم كل 5 أوزان
        if (w_idx + 1) % 5 == 0 or w_idx == len(weight_combos) - 1:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            print(f"   💾 تم حفظ التقدم بعد {w_idx+1} وزن.")

    print(f"\n✅ تم حفظ النتائج في {Path(output_file).absolute()}")

    # عرض أفضل 5 تركيبات حسب MAP
    print("\n🏆 أفضل 5 تركيبات حسب MAP:")
    sorted_weights = sorted(
        [(wkey, data['average']['map']) for wkey, data in final_results['results'].items() if 'average' in data],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for rank, (wkey, map_score) in enumerate(sorted_weights, 1):
        w = final_results['results'][wkey]['weights']
        print(f"   {rank}. {wkey} (MAP={map_score:.4f})")

    return final_results

# ============================================================
# تشغيل السكربت
# ============================================================

if __name__ == '__main__':
    # 🔥 يمكنك تعديل المعاملات هنا
    run_experiment_full(
        output_file='hybrid_full_results.json',
        top_k=10,
        step=0.2  # 0.2 = 56 تركيب، 0.3 = 36 تركيب، 0.4 = 21 تركيب
    )
