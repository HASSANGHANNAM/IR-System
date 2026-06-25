#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت لاختبار جميع التباديل الممكنة للنموذج الهجين التسلسلي (Serial Hybrid):
- فردي: نموذج واحد (Retriever فقط)
- ثنائي: نموذجين (Retriever → Re-ranker)
- ثلاثي: ثلاثة نماذج (Retriever → Re-ranker 1 → Re-ranker 2)
بدون استخدام أي أوزان.
"""

import json
import sqlite3
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm

# استيراد الدوال المطلوبة من app.py
from app import (
    get_search_assets,
    get_all_tfidf_scores,
    get_all_bm25_scores,
    get_all_bert_scores,
    load_bm25_model,
    load_bert_resources,
    compute_evaluation_metrics,
    get_qrels_for_query,
    DB_PATH,
)

# ============================================================
# 1. دوال مساعدة
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

def normalize(scores):
    """تطبيع Min-Max (اختياري، لكنه مفيد لتوحيد المقاييس عند التبديل)."""
    mn, mx = np.min(scores), np.max(scores)
    if mx - mn == 0:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def serial_search(query, model_sequence, assets, top_k=10, initial_k=100, normalize_scores=True):
    """
    تنفيذ البحث التسلسلي (Serial) لاستعلام واحد.
    
    model_sequence: قائمة بأسماء النماذج بالترتيب، مثلاً ['bm25', 'bert'].
    initial_k: عدد الوثائق المسترجعة في المرحلة الأولى.
    top_k: عدد النتائج النهائية.
    normalize_scores: إذا كان True، يتم تطبيع الدرجات في كل مرحلة (لضمان توافق المقياس).
    """
    # 1. حساب درجات جميع النماذج المطلوبة
    scores_map = {}
    if 'tfidf' in model_sequence:
        scores_map['tfidf'] = get_all_tfidf_scores(query, assets)
    if 'bm25' in model_sequence:
        scores_map['bm25'] = get_all_bm25_scores(query)
    if 'bert' in model_sequence:
        scores_map['bert'] = get_all_bert_scores(query)
    
    all_doc_ids = assets['doc_ids']
    total_docs = len(all_doc_ids)
    
    # 2. المرحلة الأولى (Retriever)
    first_model = model_sequence[0]
    first_scores = scores_map[first_model]
    
    # ترتيب جميع الوثائق حسب النموذج الأول
    sorted_indices = np.argsort(first_scores)[::-1]
    # نأخذ أفضل initial_k (أو عدد أقل إن لم يتوفر)
    current_indices = sorted_indices[:min(initial_k, total_docs)]
    current_scores = first_scores[current_indices]
    
    # 3. إذا كان التسلسل يحتوي على نموذج واحد فقط
    if len(model_sequence) == 1:
        final_indices = current_indices[:top_k]
        final_scores = current_scores[:top_k]
        return final_indices, final_scores
    
    # 4. المراحل التالية (Re-ranking)
    # نطبق التطبيع (اختياري) لتوحيد مقاييس الدرجات بين النماذج المختلفة
    if normalize_scores:
        # نطبع درجات المرحلة الحالية
        current_scores = normalize(current_scores)
    
    for i, model_name in enumerate(model_sequence[1:], start=1):
        # درجات النموذج الحالي للوثائق المتبقية فقط
        model_scores = scores_map[model_name][current_indices]
        
        # تطبيع درجات النموذج الحالي (اختياري)
        if normalize_scores:
            model_scores = normalize(model_scores)
        
        # إعادة الترتيب بناءً على درجات النموذج الحالي
        reranked_order = np.argsort(model_scores)[::-1]
        
        # تحديد عدد الوثائق المطلوب الاحتفاظ بها
        # إذا كان هذا هو آخر نموذج في التسلسل، نأخذ top_k
        # وإلا، نأخذ عدداً متوسطاً (مثلاً 50) لتقليل الحمل
        if i == len(model_sequence) - 1:
            k = min(top_k, len(current_indices))
        else:
            k = min(50, len(current_indices))
        
        # تحديث القائمة
        current_indices = current_indices[reranked_order][:k]
        current_scores = model_scores[reranked_order][:k]
    
    return current_indices, current_scores

# ============================================================
# 2. الدالة الرئيسية
# ============================================================

def run_serial_experiment(output_file='serial_permutations_results.json', top_k=10, initial_k=100):
    print("🚀 بدء تجربة التسلسلي (جميع التباديل الممكنة)...")
    print(f"   - عدد النتائج (Top-K): {top_k}")
    print(f"   - عدد النتائج الأولية (Initial-K): {initial_k}")
    
    # تحميل النماذج
    print("⏳ تحميل النماذج...")
    load_bert_resources()
    load_bm25_model()
    assets = get_search_assets('stemming')
    print("✅ تم تحميل جميع النماذج.")
    
    queries = get_all_queries()
    print(f"✅ تم تحميل {len(queries)} استعلام.")
    
    # ============================================================
    # توليد جميع التباديل الممكنة للنماذج (1, 2, 3)
    # ============================================================
    models = ['tfidf', 'bert', 'bm25']
    all_sequences = []
    
    # فردي (طول 1)
    for m in models:
        all_sequences.append([m])
    
    # ثنائي (طول 2)
    for perm in itertools.permutations(models, 2):
        all_sequences.append(list(perm))
    
    # ثلاثي (طول 3)
    for perm in itertools.permutations(models, 3):
        all_sequences.append(list(perm))
    
    print(f"✅ تم توليد {len(all_sequences)} تسلسلاً (فردي + ثنائي + ثلاثي).")
    
    # ============================================================
    # تخزين النتائج
    # ============================================================
    results_summary = {
        'top_k': top_k,
        'initial_k': initial_k,
        'total_queries': len(queries),
        'total_sequences': len(all_sequences),
        'results': []  # سيحتوي على كل استعلام × كل تسلسل
    }
    
    # ============================================================
    # المعالجة: لكل استعلام، نحسب الدرجات مرة واحدة (لتوفير الوقت)
    # ============================================================
    for qid, query_text in tqdm(queries, desc="معالجة الاستعلامات"):
        print(f"\n🔍 استعلام {qid}: {query_text[:50]}...")
        
        # جلب qrels لهذا الاستعلام (مرة واحدة)
        qrels = get_qrels_for_query(qid)
        
        # حساب درجات جميع النماذج للاستعلام الحالي (مرة واحدة)
        # (هذه هي الخطوة الأثقل، نقوم بها مرة واحدة لكل استعلام)
        all_scores = {}
        all_scores['tfidf'] = get_all_tfidf_scores(query_text, assets)
        all_scores['bm25'] = get_all_bm25_scores(query_text)
        all_scores['bert'] = get_all_bert_scores(query_text)
        
        # تطبيق كل تسلسل على هذا الاستعلام
        for seq in all_sequences:
            seq_name = " → ".join(seq)
            
            # تنفيذ البحث التسلسلي
            indices, scores = serial_search(
                query_text, 
                seq, 
                assets, 
                top_k=top_k, 
                initial_k=initial_k,
                normalize_scores=True  # تفعيل التطبيع لتحسين التوافق
            )
            
            # بناء قائمة النتائج (للتقييم)
            all_doc_ids = assets['doc_ids']
            results = []
            for idx, score in zip(indices, scores):
                if idx < len(all_doc_ids):
                    results.append({
                        'doc_id': all_doc_ids[idx],
                        'score': float(score)
                    })
            
            # حساب المقاييس
            metrics = compute_evaluation_metrics(results, qrels, top_k)
            
            # تخزين النتيجة
            results_summary['results'].append({
                'query_id': qid,
                'query_text': query_text,
                'sequence': seq,
                'sequence_name': seq_name,
                'metrics': metrics
            })
        
        # تحرير الذاكرة (حذف المصفوفات الكبيرة)
        del all_scores
        
        # حفظ مؤقت كل 10 استعلامات
        if int(qid) % 10 == 0:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=2, ensure_ascii=False)
            print(f"   💾 تم حفظ التقدم بعد الاستعلام {qid}.")
    
    # حفظ النتائج النهائية
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ تم حفظ النتائج في {Path(output_file).absolute()}")
    
    # ============================================================
    # 3. تحليل النتائج وعرض أفضل التسلسلات
    # ============================================================
    print("\n" + "="*80)
    print("📊 تحليل النتائج النهائي (التسلسلي)")
    print("="*80)
    
    # تجميع النتائج حسب اسم التسلسل
    seq_metrics = {}
    for item in results_summary['results']:
        name = item['sequence_name']
        if name not in seq_metrics:
            seq_metrics[name] = {'precision': [], 'map': [], 'ndcg': []}
        seq_metrics[name]['precision'].append(item['metrics']['precision_at_k'])
        seq_metrics[name]['map'].append(item['metrics']['map'])
        seq_metrics[name]['ndcg'].append(item['metrics']['ndcg'])
    
    # حساب المتوسطات
    avg_results = []
    for name, metrics in seq_metrics.items():
        avg_results.append({
            'name': name,
            'avg_precision': np.mean(metrics['precision']),
            'avg_map': np.mean(metrics['map']),
            'avg_ndcg': np.mean(metrics['ndcg'])
        })
    
    # الترتيب حسب MAP (تنازلياً)
    avg_results.sort(key=lambda x: x['avg_map'], reverse=True)
    
    # عرض أفضل 15 تسلسلاً
    print("\n🏆 أفضل 15 تسلسلاً حسب MAP:")
    print("-"*90)
    print(f"{'الترتيب':<6} {'التسلسل':<45} {'MAP':<10} {'P@10':<10} {'NDCG':<10}")
    print("-"*90)
    for i, res in enumerate(avg_results[:15], 1):
        name = res['name']
        if len(name) > 40:
            name = name[:37] + "..."
        print(f"{i:<6} {name:<45} {res['avg_map']:<10.4f} {res['avg_precision']:<10.4f} {res['avg_ndcg']:<10.4f}")
    
    # ============================================================
    # 4. التوصيات النهائية
    # ============================================================
    print("\n" + "="*80)
    print("🎯 التوصيات النهائية (Serial Hybrid)")
    print("="*80)
    
    # أفضل فردي
    best_individual = max([r for r in avg_results if '→' not in r['name']], key=lambda x: x['avg_map'])
    print(f"\n✅ أفضل نموذج فردي: {best_individual['name']} (MAP = {best_individual['avg_map']:.4f})")
    
    # أفضل ثنائي
    best_pair = max([r for r in avg_results if r['name'].count('→') == 1], key=lambda x: x['avg_map'], default=None)
    if best_pair:
        print(f"✅ أفضل تسلسل ثنائي: {best_pair['name']} (MAP = {best_pair['avg_map']:.4f})")
    
    # أفضل ثلاثي
    best_triple = max([r for r in avg_results if r['name'].count('→') == 2], key=lambda x: x['avg_map'], default=None)
    if best_triple:
        print(f"✅ أفضل تسلسل ثلاثي: {best_triple['name']} (MAP = {best_triple['avg_map']:.4f})")
    
    # مقارنة مع BM25 الفردي
    bm25_only = next((r for r in avg_results if r['name'] == 'bm25'), None)
    if bm25_only:
        print(f"\n📌 BM25 الفردي (المرجع): MAP = {bm25_only['avg_map']:.4f}")
    
    # الخلاصة النهائية
    print("\n💡 الخلاصة:")
    if best_individual['avg_map'] >= (best_pair['avg_map'] if best_pair else 0) and best_individual['avg_map'] >= (best_triple['avg_map'] if best_triple else 0):
        print(f"   🏆 **التسلسل الفردي ({best_individual['name']}) هو الأفضل.**")
        print(f"      لا حاجة لاستخدام Serial Hybrid في هذه المجموعة.")
    elif best_pair and best_pair['avg_map'] >= (best_triple['avg_map'] if best_triple else 0):
        print(f"   🏆 **التسلسل الثنائي ({best_pair['name']}) هو الأفضل.**")
        print(f"      استخدم هذا التسلسل في واجهة المستخدم كخيار Serial Hybrid.")
    elif best_triple:
        print(f"   🏆 **التسلسل الثلاثي ({best_triple['name']}) هو الأفضل.**")
        print(f"      استخدم هذا التسلسل في واجهة المستخدم كخيار Serial Hybrid.")
    
    # توصية إضافية
    print("\n   🔔 **توصية إضافية:**")
    if best_individual['name'] == 'bm25':
        print("   - BM25 وحده يتفوق على جميع التسلسلات الأخرى. ركز على تحسين معاملاته (k1, b).")
    else:
        print(f"   - التسلسل الأفضل هو {best_individual['name'] if best_individual['avg_map'] >= (best_pair['avg_map'] if best_pair else 0) else best_pair['name']}.")
        print("   - تأكد من تفعيل هذا التسلسل في واجهة المستخدم.")
    
    return results_summary

# ============================================================
# 5. تشغيل السكربت
# ============================================================

if __name__ == '__main__':
    run_serial_experiment(
        output_file='serial_permutations_results.json',
        top_k=10,
        initial_k=100  # عدد الوثائق المسترجعة في المرحلة الأولى
    )