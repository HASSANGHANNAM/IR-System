#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت لاختبار جميع تركيبات معاملات BM25 (k1, b) على جميع الاستعلامات
لتحديد أفضل الإعدادات الافتراضية.
النتائج: تُحفظ في ملف JSON وتُعرض على الكونسول.
"""

# ============================================================
# 🔥 إضافة الدوال المفقودة للـ pickle (لتجنب AttributeError)
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
import numpy as np
from pathlib import Path
from tqdm import tqdm
from itertools import product

from app import (
    get_search_assets,
    get_all_bm25_scores,
    load_bm25_model,
    compute_evaluation_metrics,
    get_qrels_for_query,
    DB_PATH,
)

# ============================================================
# دوال مساعدة
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

def evaluate_bm25_with_params(query, k1, b, assets, top_k=10):
    """
    حساب نتائج BM25 باستخدام معاملات محددة (k1, b) لاستعلام واحد.
    يتم تعديل معاملات النموذج مؤقتاً، ثم استعادتها بعد الحساب.
    """
    global bm25_model
    from app import bm25_model as global_bm25
    bm25_model = global_bm25
    
    if bm25_model is None:
        load_bm25_model()
    
    # حفظ المعاملات الحالية
    original_k1 = bm25_model.k1
    original_b = bm25_model.b
    
    # تطبيق المعاملات الجديدة
    bm25_model.k1 = k1
    bm25_model.b = b
    
    # حساب الدرجات
    scores = get_all_bm25_scores(query)
    
    # استعادة المعاملات الأصلية
    bm25_model.k1 = original_k1
    bm25_model.b = original_b
    
    return scores

def run_experiment(output_file='bm25_params_results.json', top_k=10):
    """تشغيل التجربة على جميع الاستعلامات وجميع تركيبات المعاملات."""
    print("="*80)
    print("🚀 بدء تجربة معاملات BM25...")
    print("="*80)
    print(f"   - عدد النتائج (Top-K): {top_k}")
    
    # تحميل النماذج
    print("\n⏳ تحميل النماذج...")
    load_bm25_model()
    assets = get_search_assets('stemming')
    print("✅ تم تحميل النماذج.")
    
    queries = get_all_queries()
    print(f"✅ تم تحميل {len(queries)} استعلام.")
    
    # توليد تركيبات (k1, b)
    k1_values = np.arange(0.5, 3.1, 0.5).round(1).tolist()  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    b_values = np.arange(0.0, 1.05, 0.1).round(2).tolist()   # [0.0, 0.1, ..., 1.0]
    total_combinations = len(k1_values) * len(b_values)
    print(f"✅ تم توليد {total_combinations} تركيبة (k1, b).")
    
    # هيكل النتائج
    results = []
    all_doc_ids = assets['doc_ids']
    
    # تجربة كل تركيبة
    print("\n⏳ جاري اختبار المعاملات...")
    with tqdm(total=total_combinations, desc="اختبار المعاملات", unit="تركيبة") as pbar:
        for k1 in k1_values:
            for b in b_values:
                all_metrics = []
                failed_queries = 0
                
                for qid, query_text in queries:
                    try:
                        # حساب درجات BM25 مع المعاملات الحالية
                        scores = evaluate_bm25_with_params(query_text, k1, b, assets, top_k)
                        
                        # ترتيب النتائج
                        top_indices = np.argsort(scores)[::-1][:top_k]
                        results_list = []
                        for idx in top_indices:
                            if idx < len(all_doc_ids):
                                results_list.append({
                                    'doc_id': all_doc_ids[idx],
                                    'score': float(scores[idx])
                                })
                        
                        # جلب qrels
                        qrels = get_qrels_for_query(qid)
                        
                        # حساب المقاييس
                        metrics = compute_evaluation_metrics(results_list, qrels, top_k)
                        all_metrics.append(metrics)
                    except Exception as e:
                        failed_queries += 1
                        continue
                
                if all_metrics:
                    avg_precision = np.mean([m['precision_at_k'] for m in all_metrics])
                    avg_map = np.mean([m['map'] for m in all_metrics])
                    avg_ndcg = np.mean([m['ndcg'] for m in all_metrics])
                    
                    results.append({
                        'k1': k1,
                        'b': b,
                        'avg_precision': round(avg_precision, 6),
                        'avg_map': round(avg_map, 6),
                        'avg_ndcg': round(avg_ndcg, 6),
                        'queries_processed': len(all_metrics),
                        'failed_queries': failed_queries
                    })
                else:
                    results.append({
                        'k1': k1,
                        'b': b,
                        'avg_precision': 0.0,
                        'avg_map': 0.0,
                        'avg_ndcg': 0.0,
                        'queries_processed': 0,
                        'failed_queries': len(queries)
                    })
                
                pbar.update(1)
    
    # ترتيب النتائج حسب MAP (تنازلياً)
    results.sort(key=lambda x: x['avg_map'], reverse=True)
    
    # حفظ النتائج في ملف JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ تم حفظ النتائج في {output_path.absolute()}")
    
    # ============================================================
    # عرض النتائج على الكونسول
    # ============================================================
    print("\n" + "="*80)
    print("📊 نتائج تجربة معاملات BM25")
    print("="*80)
    
    # عرض أفضل 10 تركيبات
    print("\n🏆 أفضل 10 تركيبات (k1, b) حسب MAP:")
    print("-"*80)
    print(f"{'الترتيب':<6} {'k1':<10} {'b':<10} {'MAP':<12} {'P@10':<12} {'NDCG':<12} {'المعالجة':<10}")
    print("-"*80)
    for i, res in enumerate(results[:10], 1):
        processed = f"{res['queries_processed']}/{len(queries)}"
        print(f"{i:<6} {res['k1']:<10.1f} {res['b']:<10.2f} {res['avg_map']:<12.4f} {res['avg_precision']:<12.4f} {res['avg_ndcg']:<12.4f} {processed:<10}")
    
    # عرض أفضل 3 تركيبات بشكل مميز
    print("\n" + "="*80)
    print("🥇 أفضل 3 تركيبات:")
    print("="*80)
    for i, res in enumerate(results[:3], 1):
        medal = ['🥇', '🥈', '🥉'][i-1]
        print(f"\n{medal} الترتيب {i}:")
        print(f"   • k1 = {res['k1']:.1f}")
        print(f"   • b  = {res['b']:.2f}")
        print(f"   • MAP = {res['avg_map']:.4f}")
        print(f"   • Precision@10 = {res['avg_precision']:.4f}")
        print(f"   • NDCG = {res['avg_ndcg']:.4f}")
    
    # ============================================================
    # التوصية النهائية
    # ============================================================
    print("\n" + "="*80)
    print("🎯 التوصية النهائية")
    print("="*80)
    
    best = results[0]
    print(f"\n✅ أفضل معاملات BM25 حسب التجربة:")
    print(f"   • k1 = {best['k1']:.1f}")
    print(f"   • b  = {best['b']:.2f}")
    print(f"   • MAP = {best['avg_map']:.4f} (على {len(queries)} استعلام)")
    
    # البحث عن القيم الافتراضية الحالية (1.5, 0.75)
    default_result = next((r for r in results if r['k1'] == 1.5 and r['b'] == 0.75), None)
    if default_result:
        print(f"\n📌 القيم الافتراضية الحالية (k1=1.5, b=0.75):")
        print(f"   • MAP = {default_result['avg_map']:.4f}")
        diff = best['avg_map'] - default_result['avg_map']
        if diff > 0:
            print(f"   • التحسين = +{diff:.4f} ({(diff/default_result['avg_map']*100):.2f}%)")
        else:
            print(f"   • الفرق = {diff:.4f} (القيم الافتراضية أفضل أو متساوية)")
    
    # عرض الكود المطلوب لتحديث الواجهة
    print(f"\n📝 لتحديث القيم الافتراضية في الواجهة (index.html):")
    print(f"   <input type=\"number\" id=\"bm25K1\" value=\"{best['k1']:.1f}\" ...>")
    print(f"   <input type=\"number\" id=\"bm25B\" value=\"{best['b']:.2f}\" ...>")
    
    print(f"\n📝 لتحديث زر 'The Best' في script.js:")
    print(f'   bm25K1.value = "{best["k1"]:.1f}";')
    print(f'   bm25B.value = "{best["b"]:.2f}";')
    
    print("\n" + "="*80)
    print("✅ انتهت التجربة بنجاح!")
    print("="*80)
    
    return results

# ============================================================
# تشغيل السكربت
# ============================================================

if __name__ == '__main__':
    run_experiment(
        output_file='bm25_params_results.json',
        top_k=10
    )