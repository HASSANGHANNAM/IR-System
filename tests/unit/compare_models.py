#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
مقارنة الأداء والجودة بين النماذج الثلاثة:
- TF-IDF
- BM25
- BERT

يقوم السكربت بـ:
1. قراءة جميع الاستعلامات من queries_cleaned.jsonl.
2. إرسال كل استعلام إلى الخادم المحلي مع model مختلف و evaluate=True.
3. حساب متوسط زمن الاستجابة، MAP، NDCG، Precision@K لكل نموذج.
4. عرض النتائج في جدول مقارن.
"""

import json
import time
import statistics
import requests
from tabulate import tabulate  # اختياري، لكن مفيد للعرض

# ===== الإعدادات =====
API_URL = "http://localhost:5000/api/search"
QUERIES_FILE = "data/processed_stemming/queries_cleaned.jsonl"
TOP_K = 10
MODELS = ["tfidf", "bm25", "bert"]

# ===== قراءة الاستعلامات =====
queries = []  # قائمة من (query_id, query_text)
with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        qid = data.get('query_id')
        text = data.get('original_text') or data.get('text')
        if qid and text:
            queries.append((qid, text))

print(f"✅ تم تحميل {len(queries)} استعلام.")

# ===== دالة لإرسال طلب بحث =====
def search(query_id, query_text, model):
    payload = {
        "query": query_text,
        "query_id": query_id,
        "model": model,
        "top_k": TOP_K,
        "refine": False,
        "preprocessing": "stemming",
        "evaluate": True
    }
    start = time.time()
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        end = time.time()
        if resp.status_code == 200:
            data = resp.json()
            metrics = data.get('metrics', {})
            return {
                "time": end - start,
                "precision": metrics.get('precision_at_k', 0),
                "map": metrics.get('map', 0),
                "ndcg": metrics.get('ndcg', 0),
                "success": True
            }
        else:
            print(f"⚠️ خطأ في الاستعلام {query_id} (النموذج {model}): {resp.status_code}")
            return None
    except Exception as e:
        print(f"❌ فشل الاتصال: {e}")
        return None

# ===== تجربة النماذج =====
results = {model: {"times": [], "precision": [], "map": [], "ndcg": []} for model in MODELS}

for model in MODELS:
    print(f"\n⚡ جاري اختبار النموذج: {model.upper()}")
    for idx, (qid, qtext) in enumerate(queries, 1):
        print(f"   {idx}/{len(queries)}", end="\r")
        res = search(qid, qtext, model)
        if res and res["success"]:
            results[model]["times"].append(res["time"])
            results[model]["precision"].append(res["precision"])
            results[model]["map"].append(res["map"])
            results[model]["ndcg"].append(res["ndcg"])
    print(f"✅ {model.upper()} اكتمل: {len(results[model]['times'])}/{len(queries)} طلب ناجح.")

# ===== حساب الإحصائيات =====
stats = {}
for model in MODELS:
    times = results[model]["times"]
    if not times:
        continue
    stats[model] = {
        "avg_time": statistics.mean(times),
        "min_time": min(times),
        "max_time": max(times),
        "avg_precision": statistics.mean(results[model]["precision"]),
        "avg_map": statistics.mean(results[model]["map"]),
        "avg_ndcg": statistics.mean(results[model]["ndcg"]),
    }

# ===== عرض النتائج =====
print("\n" + "="*80)
print("📊 مقارنة النماذج (متوسط الأداء على جميع الاستعلامات)")
print("="*80)

headers = ["النموذج", "الوقت (ث)", "Precision@K", "MAP", "NDCG"]
rows = []
for model in MODELS:
    if model in stats:
        s = stats[model]
        rows.append([
            model.upper(),
            f"{s['avg_time']:.3f}",
            f"{s['avg_precision']*100:.2f}%",
            f"{s['avg_map']*100:.2f}%",
            f"{s['avg_ndcg']*100:.2f}%"
        ])
    else:
        rows.append([model.upper(), "—", "—", "—", "—"])

# طباعة جدول باستخدام tabulate (إن وجدت) أو يدوياً
try:
    print(tabulate(rows, headers=headers, tablefmt="grid"))
except ImportError:
    # fallback: طباعة بسيطة
    print(" | ".join(headers))
    print("-" * 60)
    for row in rows:
        print(" | ".join(row))

# ===== حفظ النتائج في ملف JSON =====
output = {
    "top_k": TOP_K,
    "queries_count": len(queries),
    "stats": stats
}
with open("comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\n✅ تم حفظ النتائج في comparison_results.json")

print("\n🏁 انتهى السكربت.")