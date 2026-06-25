#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت لتجربة جميع الأوزان الممكنة للنموذج الهجين المتوازي (Parallel Hybrid)
على جميع الاستعلامات، وحساب المقاييس، وتخزين النتائج في ملف JSON.
"""

import json
import sqlite3
import time
from pathlib import Path
from itertools import product
import numpy as np

# استيراد الدوال الأساسية من app.py (نفترض أنه في نفس المجلد)
from app import (
    clean_query,
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
    ROOT,
)

# ============================================================
# 1. تحميل جميع الاستعلامات من قاعدة البيانات
# ============================================================

def get_all_queries():
    """تعيد قائمة بجميع الاستعلامات من قاعدة البيانات (query_id, text)."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT query_id, text FROM queries ORDER BY CAST(query_id AS INTEGER)")
        rows = cursor.fetchall()
        return [(str(qid), text) for qid, text in rows if text and text.strip()]
    finally:
        conn.close()

# ============================================================
# 2. دوال مساعدة لتجربة الأوزان
# ============================================================

def generate_weight_combinations(step=0.1, min_weight=0.0, include_zero=False):
    """
    توليد جميع تركيبات الأوزان (tfidf, bert, bm25) التي مجموعها = 1.0.
    step: خطوة التغيير (0.1 تعني 0.0, 0.1, 0.2, ..., 1.0).
    min_weight: أقل وزن مسموح (لتجنب الأوزان الصغيرة جداً).
    include_zero: إذا كان True، يسمح بوزن = 0 (لاختبار استبعاد نموذج).
    """
    weights = np.arange(min_weight, 1.0 + step, step).round(2).tolist()
    if not include_zero:
        weights = [w for w in weights if w > 0]
    
    combinations = []
    for w_tfidf in weights:
        for w_bert in weights:
            for w_bm25 in weights:
                if abs(w_tfidf + w_bert + w_bm25 - 1.0) < 0.001:
                    # نضمن أن الأوزان ليست كلها صفر
                    if w_tfidf > 0 or w_bert > 0 or w_bm25 > 0:
                        combinations.append({
                            'tfidf': round(w_tfidf, 2),
                            'bert': round(w_bert, 2),
                            'bm25': round(w_bm25, 2)
                        })
    return combinations

def evaluate_hybrid_parallel(query, weights, top_k=10, assets=None):
    """
    تنفيذ البحث الهجين المتوازي لاستعلام واحد، وإرجاع النتائج والمقاييس.
    """
    if assets is None:
        assets = get_search_assets('stemming')
    
    # 1. حساب درجات كل نموذج
    scores_tfidf = get_all_tfidf_scores(query, assets)
    scores_bm25 = get_all_bm25_scores(query)
    scores_bert = get_all_bert_scores(query)
    
    # 2. تطبيع Min-Max
    def normalize(scores):
        min_s = np.min(scores)
        max_s = np.max(scores)
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    norm_tfidf = normalize(scores_tfidf)
    norm_bm25 = normalize(scores_bm25)
    norm_bert = normalize(scores_bert)
    
    # 3. الدمج المرجح
    final_scores = (weights['tfidf'] * norm_tfidf +
                    weights['bert'] * norm_bert +
                    weights['bm25'] * norm_bm25)
    
    # 4. ترتيب النتائج
    all_doc_ids = assets['doc_ids']
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        if idx < len(all_doc_ids):
            results.append({
                'doc_id': all_doc_ids[idx],
                'score': float(final_scores[idx])
            })
    
    # إضافة النصوص (للتقييم فقط)
    texts = get_documents_texts([item['doc_id'] for item in results])
    for item in results:
        item['text'] = texts.get(str(item['doc_id']), '')
    
    return results

# ============================================================
# 3. الدالة الرئيسية
# ============================================================

def run_experiment(output_file='hybrid_weights_results.json', top_k=10, step=0.2):
    """
    تشغيل التجربة على جميع الاستعلامات وجميع تركيبات الأوزان.
    """
    print("🚀 بدء تجربة الأوزان للنموذج الهجين المتوازي...")
    print(f"   - عدد النتائج (Top-K): {top_k}")
    print(f"   - خطوة الأوزان: {step}")
    
    # تحميل النماذج مسبقاً (مرة واحدة)
    print("⏳ تحميل النماذج...")
    load_bert_resources()
    load_bm25_model()
    assets = get_search_assets('stemming')
    print("✅ تم تحميل جميع النماذج.")
    
    # تحميل الاستعلامات
    queries = get_all_queries()
    print(f"✅ تم تحميل {len(queries)} استعلام.")
    
    if not queries:
        print("⚠️ لا توجد استعلامات! تأكد من قاعدة البيانات.")
        return
    
    # توليد تركيبات الأوزان
    weight_combos = generate_weight_combinations(step=step, include_zero=True)
    print(f"✅ تم توليد {len(weight_combos)} تركيبة أوزان.")
    
    # هيكل النتائج
    results = {
        'top_k': top_k,
        'step': step,
        'total_queries': len(queries),
        'total_weights': len(weight_combos),
        'results': {}  # weight_key -> { query_id: metrics }
    }
    
    # التجربة
    total_combos = len(weight_combos)
    for idx, weights in enumerate(weight_combos, 1):
        weight_key = f"tfidf{weights['tfidf']:.2f}_bert{weights['bert']:.2f}_bm25{weights['bm25']:.2f}"
        print(f"\n🔍 [{idx}/{total_combos}] اختبار الأوزان: {weight_key}")
        
        results['results'][weight_key] = {}
        results['results'][weight_key]['weights'] = weights
        results['results'][weight_key]['queries'] = {}
        
        # حساب المقاييس لكل استعلام
        for qid, query_text in queries:
            try:
                # البحث
                search_results = evaluate_hybrid_parallel(query_text, weights, top_k, assets)
                
                # جلب qrels لهذا الاستعلام
                relevant_docs = get_qrels_for_query(qid)
                
                # حساب المقاييس
                metrics = compute_evaluation_metrics(search_results, relevant_docs, top_k)
                
                # تخزين النتيجة
                results['results'][weight_key]['queries'][qid] = {
                    'query_text': query_text,
                    'metrics': metrics,
                    'num_results': len(search_results)
                }
            except Exception as e:
                print(f"   ⚠️ خطأ في الاستعلام {qid}: {e}")
                results['results'][weight_key]['queries'][qid] = {
                    'query_text': query_text,
                    'error': str(e)
                }
        
        # حساب المتوسطات لهذه الأوزان
        all_metrics = [q['metrics'] for q in results['results'][weight_key]['queries'].values() if 'metrics' in q]
        if all_metrics:
            avg_precision = np.mean([m['precision_at_k'] for m in all_metrics])
            avg_map = np.mean([m['map'] for m in all_metrics])
            avg_ndcg = np.mean([m['ndcg'] for m in all_metrics])
            results['results'][weight_key]['average'] = {
                'precision_at_k': round(avg_precision, 6),
                'map': round(avg_map, 6),
                'ndcg': round(avg_ndcg, 6)
            }
            # طباعة المتوسطات على الكونسول
            print(f"   📊 المتوسط: P@{top_k}={avg_precision:.4f}, MAP={avg_map:.4f}, NDCG={avg_ndcg:.4f}")
    
    # حفظ النتائج في ملف JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ تم حفظ النتائج في {output_path.absolute()}")
    
    # طباعة أفضل 5 تركيبات حسب MAP
    print("\n🏆 أفضل 5 تركيبات حسب MAP:")
    sorted_weights = sorted(
        [(wkey, data['average']['map']) for wkey, data in results['results'].items() if 'average' in data],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for rank, (wkey, map_score) in enumerate(sorted_weights, 1):
        w = results['results'][wkey]['weights']
        print(f"   {rank}. {wkey} (MAP={map_score:.4f})")
    
    return results

# ============================================================
# 4. تشغيل السكربت
# ============================================================

if __name__ == '__main__':
    # يمكنك تغيير المعاملات هنا
    run_experiment(
        output_file='hybrid_weights_results.json',
        top_k=10,
        step=0.2  # خطوة الأوزان (0.2 تعطي تركيبات: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )