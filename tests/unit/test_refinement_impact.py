#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت لتقييم تأثير Query Refinement (تصحيح إملائي + مرادفات) على جميع النماذج.
يقوم بتشغيل البحث لكل استعلام (49) مرتين: بدون تحسين و مع تحسين.
ويحسب متوسط المقاييس (MAP, Precision@10, NDCG) لكل نموذج.
"""

# ============================================================
# 🔥 تعريف الدوال المفقودة للـ pickle (لتجنب AttributeError)
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
from collections import defaultdict

from app import (
    get_search_assets,
    search_tfidf,
    search_bm25,
    search_bert,
    search_hybrid,
    load_bm25_model,
    load_bert_resources,
    compute_evaluation_metrics,
    get_qrels_for_query,
    refine_query_text,
    DB_PATH,
)

# ============================================================
# إعدادات التجربة
# ============================================================
TOP_K = 10
OUTPUT_FILE = "refinement_impact_results.json"

# أفضل الأوزان للهجين المتوازي من تجاربك السابقة
BEST_PARALLEL_WEIGHTS = {
    'tfidf': 0.20,
    'bert': 0.00,
    'bm25': 0.80
}

# أفضل معاملات BM25 من تجاربك السابقة
BM25_K1 = 2.0
BM25_B = 0.60

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

def run_search_for_query(query_text, model, assets, refine=False, **kwargs):
    """
    تنفيذ البحث لاستعلام واحد باستخدام النموذج المحدد.
    """
    # تطبيق التحسين إذا كان مطلوباً
    if refine:
        query_text = refine_query_text(query_text)['expanded']
    
    # اختيار دالة البحث المناسبة
    if model == 'tfidf':
        return search_tfidf(query_text, assets, TOP_K)
    elif model == 'bm25':
        return search_bm25(query_text, TOP_K)
    elif model == 'bert':
        return search_bert(query_text, top_k=TOP_K, use_title=True, alpha=0.4)
    elif model == 'hybrid_parallel':
        return search_hybrid(query_text, 'parallel', BEST_PARALLEL_WEIGHTS, TOP_K, 'stemming')
    elif model == 'hybrid_serial':
        return search_hybrid(query_text, 'serial', None, TOP_K, 'stemming')
    else:
        raise ValueError(f"Model {model} not supported")

# ============================================================
# الدالة الرئيسية
# ============================================================

def run_experiment():
    print("="*80)
    print("🚀 بدء تجربة تأثير Query Refinement على جميع النماذج...")
    print("="*80)
    print(f"   - عدد النتائج (Top-K): {TOP_K}")
    print(f"   - عدد النماذج: 5 (TF-IDF, BM25, BERT, Hybrid Parallel, Hybrid Serial)")
    
    # تحميل النماذج والموارد (مرة واحدة)
    print("\n⏳ تحميل النماذج والموارد...")
    load_bm25_model()
    load_bert_resources()
    assets = get_search_assets('stemming')
    print("✅ تم تحميل جميع النماذج.")
    
    # تحميل الاستعلامات
    queries = get_all_queries()
    print(f"✅ تم تحميل {len(queries)} استعلام.")
    
    # قائمة النماذج المراد اختبارها
    models = ['tfidf', 'bm25', 'bert', 'hybrid_parallel', 'hybrid_serial']
    
    # هيكل النتائج
    results_summary = {
        'top_k': TOP_K,
        'models': {},
        'total_queries': len(queries)
    }
    
    # ============================================================
    # المعالجة
    # ============================================================
    for model_name in tqdm(models, desc="معالجة النماذج"):
        print(f"\n🔍 معالجة النموذج: {model_name}")
        
        # تخزين النتائج المؤقتة لهذا النموذج
        model_data = {
            'without_refinement': {'metrics': []},
            'with_refinement': {'metrics': []}
        }
        
        # تفاصيل لكل استعلام (للحفظ في JSON)
        query_details = {}
        
        for qid, query_text in tqdm(queries, desc=f"  الاستعلامات ({model_name})", leave=False):
            try:
                qrels = get_qrels_for_query(qid)
                query_details[qid] = {'query_text': query_text}
                
                # 1. البحث بدون تحسين
                results_no_ref = run_search_for_query(query_text, model_name, assets, refine=False)
                metrics_no_ref = compute_evaluation_metrics(results_no_ref, qrels, TOP_K)
                model_data['without_refinement']['metrics'].append(metrics_no_ref)
                query_details[qid]['without_refinement'] = metrics_no_ref
                
                # 2. البحث مع تحسين
                results_ref = run_search_for_query(query_text, model_name, assets, refine=True)
                metrics_ref = compute_evaluation_metrics(results_ref, qrels, TOP_K)
                model_data['with_refinement']['metrics'].append(metrics_ref)
                query_details[qid]['with_refinement'] = metrics_ref
                
            except Exception as e:
                print(f"      ⚠️ خطأ في الاستعلام {qid} مع النموذج {model_name}: {e}")
                query_details[qid] = {'error': str(e)}
                continue
        
        # حساب المتوسطات لهذا النموذج
        avg_results = {}
        for key in ['without_refinement', 'with_refinement']:
            metrics_list = model_data[key]['metrics']
            if metrics_list:
                avg_results[key] = {
                    'avg_precision': np.mean([m['precision_at_k'] for m in metrics_list]),
                    'avg_map': np.mean([m['map'] for m in metrics_list]),
                    'avg_ndcg': np.mean([m['ndcg'] for m in metrics_list])
                }
            else:
                avg_results[key] = {'avg_precision': 0, 'avg_map': 0, 'avg_ndcg': 0}
        
        # حساب التحسين (فرق)
        improvement = {}
        if 'without_refinement' in avg_results and 'with_refinement' in avg_results:
            improvement = {
                'precision_diff': avg_results['with_refinement']['avg_precision'] - avg_results['without_refinement']['avg_precision'],
                'map_diff': avg_results['with_refinement']['avg_map'] - avg_results['without_refinement']['avg_map'],
                'ndcg_diff': avg_results['with_refinement']['avg_ndcg'] - avg_results['without_refinement']['avg_ndcg']
            }
        
        # تخزين النتائج في المخطط الرئيسي
        results_summary['models'][model_name] = {
            'queries': query_details,
            'averages': avg_results,
            'improvement': improvement,
            'num_queries_processed': len(model_data['without_refinement']['metrics'])
        }
        
        # عرض نتائج هذا النموذج على الكونسول فوراً
        print(f"\n📊 النتائج المتوسطة للنموذج: {model_name}")
        if 'without_refinement' in avg_results:
            no_ref = avg_results['without_refinement']
            ref = avg_results['with_refinement']
            print(f"   بدون تحسين: MAP={no_ref['avg_map']:.4f}, P@10={no_ref['avg_precision']:.4f}, NDCG={no_ref['avg_ndcg']:.4f}")
            print(f"   مع تحسين:   MAP={ref['avg_map']:.4f}, P@10={ref['avg_precision']:.4f}, NDCG={ref['avg_ndcg']:.4f}")
            print(f"   التحسين:    MAP={improvement['map_diff']:.4f}, P@10={improvement['precision_diff']:.4f}, NDCG={improvement['ndcg_diff']:.4f}")
        else:
            print("   ⚠️ لم يتم معالجة أي استعلام لهذا النموذج.")
    
    # حفظ النتائج في ملف JSON
    output_path = Path(OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ تم حفظ النتائج في {output_path.absolute()}")
    
    # ============================================================
    # عرض جدول المقارنة النهائي (ملخص)
    # ============================================================
    print("\n" + "="*80)
    print("📊 ملخص تأثير Query Refinement على جميع النماذج")
    print("="*80)
    print(f"{'النموذج':<18} {'MAP (بدون)':<12} {'MAP (مع)':<12} {'التحسين':<12} {'P@10 (بدون)':<12} {'P@10 (مع)':<12}")
    print("-"*90)
    
    for model_name in models:
        if model_name not in results_summary['models']:
            continue
        data = results_summary['models'][model_name]
        if 'averages' not in data or 'without_refinement' not in data['averages']:
            continue
        
        no_ref = data['averages']['without_refinement']
        ref = data['averages']['with_refinement']
        imp = data['improvement']
        
        # تنسيق الأسماء للعرض
        display_name = model_name.replace('_', ' ').title()
        if model_name == 'hybrid_parallel':
            display_name = 'Hybrid (Parallel)'
        elif model_name == 'hybrid_serial':
            display_name = 'Hybrid (Serial)'
        
        print(f"{display_name:<18} {no_ref['avg_map']:<12.4f} {ref['avg_map']:<12.4f} {imp['map_diff']:<+12.4f} {no_ref['avg_precision']:<12.4f} {ref['avg_precision']:<12.4f}")
    
    # التوصية النهائية
    print("\n" + "="*80)
    print("🎯 التوصية النهائية")
    print("="*80)
    
    # تحديد النموذج الأكثر استفادة من التحسين (حسب تحسين MAP)
    best_improvement = None
    best_model = None
    for model_name in models:
        if model_name not in results_summary['models']:
            continue
        data = results_summary['models'][model_name]
        if 'improvement' in data:
            imp = data['improvement']
            if best_improvement is None or imp['map_diff'] > best_improvement:
                best_improvement = imp['map_diff']
                best_model = model_name
    
    if best_model:
        display_name = best_model.replace('_', ' ').title()
        if best_model == 'hybrid_parallel':
            display_name = 'Hybrid (Parallel)'
        elif best_model == 'hybrid_serial':
            display_name = 'Hybrid (Serial)'
        print(f"✅ النموذج الأكثر استفادة من التحسين: {display_name} (تحسين MAP = {best_improvement:.4f})")
    else:
        print("⚠️ لم يتم العثور على بيانات كافية للتوصية.")
    
    print("\n💡 ملاحظة: التحسين الحالي (تصحيح إملائي + مرادفات) قد يحسن الأداء بشكل طفيف.")
    print("   النتائج أعلاه ستظهر الفرق الفعلي بالأرقام.")
    
    return results_summary

# ============================================================
# تشغيل السكربت
# ============================================================

if __name__ == '__main__':
    run_experiment()