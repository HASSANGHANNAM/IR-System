import json
import time
import requests
import statistics

# المسار إلى ملف الكويريات المنظفة
QUERIES_FILE = "data/processed_stemming/queries_cleaned.jsonl"
API_URL = "http://localhost:5000/api/search"

# قراءة الكويريات
queries = []
with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # استخدم النص الأصلي (original_text) إن وجد، وإلا استخدم النص المنظف (text)
        if 'original_text' in data and data['original_text']:
            queries.append(data['original_text'])
        else:
            queries.append(data.get('text', ''))

print(f"✅ تم تحميل {len(queries)} كويري.")

# دالة لقياس زمن طلب واحد
def measure_time(query, model, top_k=10):
    payload = {
        "query": query,
        "model": model,
        "top_k": top_k,
        "refine": False,
        "preprocessing": "stemming",
        "evaluate": False  # تعطيل التقييم لسرعة الاختبار
    }
    start = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        end = time.time()
        if response.status_code == 200:
            return end - start
        else:
            print(f"⚠️ خطأ في الاستعلام '{query[:30]}...': {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ فشل الاتصال: {e}")
        return None

# اختبار النموذجين
models = ["tfidf", "bert"]
results = {}

for model in models:
    print(f"\n⚡ جاري اختبار نموذج: {model.upper()}")
    times = []
    total = len(queries)
    for i, q in enumerate(queries, 1):
        print(f"   تنفيذ {i}/{total}...", end="\r")
        t = measure_time(q, model)
        if t is not None:
            times.append(t)
    # حساب الإحصائيات
    if times:
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        results[model] = {
            "avg": avg_time,
            "min": min_time,
            "max": max_time,
            "median": median_time,
            "count": len(times)
        }
        print(f"\n✅ نموذج {model.upper()} اكتمل: {len(times)}/{total} طلب ناجح.")
    else:
        print(f"\n❌ نموذج {model.upper()} لم يستجب لأي طلب.")

# عرض النتائج النهائية
print("\n" + "="*50)
print("📊 مقارنة سرعة النماذج (بالثواني)")
print("="*50)
for model, stats in results.items():
    print(f"\n🔹 {model.upper()}:")
    print(f"   متوسط الوقت: {stats['avg']:.4f} ثانية")
    print(f"   أسرع طلب:   {stats['min']:.4f} ثانية")
    print(f"   أبطأ طلب:   {stats['max']:.4f} ثانية")
    print(f"   الوسيط:     {stats['median']:.4f} ثانية")
    print(f"   الناجحة:    {stats['count']}/49")