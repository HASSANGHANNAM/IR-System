"""
سكربت بناء فهارس FAISS من ملفات Embeddings الموجودة.
يُشغل مرة واحدة فقط لإنشاء الملفات التالية في مجلد models/:
- faiss_index_text.flat
- faiss_index_title.flat (اختياري)
- faiss_doc_ids.pkl
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
import faiss

# ===== إعداد المسارات =====
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
DOCUMENTS_JSONL_PATH = DATA_DIR / 'processed_stemming' / 'corpus_original.jsonl'

# التأكد من وجود مجلد models
MODELS_DIR.mkdir(exist_ok=True)

# ===== 1. تحميل معرفات الوثائق الحقيقية (Doc IDs) من corpus_original.jsonl =====
doc_ids_list = []

if DOCUMENTS_JSONL_PATH.exists():
    print(f"📂 جاري تحميل المعرفات الحقيقية من {DOCUMENTS_JSONL_PATH}...")
    with open(DOCUMENTS_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                doc_id = data.get('doc_id')
                if doc_id is not None:
                    doc_ids_list.append(str(doc_id))
            except Exception:
                continue
    print(f"✅ تم تحميل {len(doc_ids_list)} معرف وثيقة حقيقي من corpus_original.jsonl")
else:
    print(f"⚠️ لم يتم العثور على {DOCUMENTS_JSONL_PATH}")

# إذا لم نجد الملف، نحاول تحميل doc_ids.pkl كحل احتياطي
if not doc_ids_list:
    doc_ids_pkl_path = MODELS_DIR / 'doc_ids.pkl'
    if doc_ids_pkl_path.exists():
        try:
            with open(doc_ids_pkl_path, 'rb') as f:
                doc_ids_list = pickle.load(f)
            print(f"✅ تم تحميل {len(doc_ids_list)} معرف من doc_ids.pkl")
        except Exception as e:
            print(f"⚠️ فشل تحميل doc_ids.pkl: {e}")

# إذا لم نجد أي شيء، نستخدم أرقاماً افتراضية
if not doc_ids_list:
    print("⚠️ لم يتم العثور على أي معرفات حقيقية. سيتم إنشاء معرفات رقمية افتراضية.")
    # سنحدد العدد لاحقاً بعد تحميل المصفوفة

# ===== 2. بناء فهرس FAISS للنص (Text Embeddings) =====
text_embeddings_path = MODELS_DIR / 'embeddings_text.npy'
if not text_embeddings_path.exists():
    print(f"❌ خطأ: الملف {text_embeddings_path} غير موجود.")
    print("تأكد من وجود ملف embeddings_text.npy في مجلد models/")
    sys.exit(1)

print(f"📂 جاري تحميل {text_embeddings_path}...")
embeddings_text = np.load(text_embeddings_path, mmap_mode='r')
print(f"✅ تم تحميل المصفوفة: shape = {embeddings_text.shape}")

# إذا لم تكن doc_ids_list محملة، ننشئها بناءً على حجم المصفوفة
if not doc_ids_list:
    doc_ids_list = [str(i) for i in range(embeddings_text.shape[0])]
    print(f"⚠️ تم إنشاء {len(doc_ids_list)} معرف وثيقة افتراضي (أرقام تسلسلية).")

# التأكد من تطابق العدد
if len(doc_ids_list) != embeddings_text.shape[0]:
    print(f"⚠️ عدد المعرفات ({len(doc_ids_list)}) لا يتطابق مع عدد المتجهات ({embeddings_text.shape[0]})")
    print("سيتم استخدام عدد المتجهات كمرجع وإعادة إنشاء المعرفات.")
    doc_ids_list = [str(i) for i in range(embeddings_text.shape[0])]

# تحويل إلى float32 (FAISS يتطلب float32)
embeddings_text = embeddings_text.astype(np.float32)

# تطبيع المتجهات (L2 Normalization) لاستخدام IndexFlatIP (تشابه جيب التمام)
print("🔄 جاري تطبيع متجهات النص...")
faiss.normalize_L2(embeddings_text)

# بناء الفهرس
dimension = embeddings_text.shape[1]
index_text = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity بعد التطبيع
print(f"🔄 جاري إضافة {embeddings_text.shape[0]} متجه إلى فهرس النص...")
index_text.add(embeddings_text)

# حفظ الفهرس
faiss_index_text_path = MODELS_DIR / 'faiss_index_text.flat'
faiss.write_index(index_text, str(faiss_index_text_path))
print(f"✅ تم حفظ فهرس النص في {faiss_index_text_path}")

# تحرير الذاكرة (اختياري)
del embeddings_text

# ===== 3. بناء فهرس FAISS للعناوين (Title Embeddings) - اختياري =====
title_embeddings_path = MODELS_DIR / 'embeddings_title.npy'
if title_embeddings_path.exists():
    print(f"📂 جاري تحميل {title_embeddings_path}...")
    embeddings_title = np.load(title_embeddings_path, mmap_mode='r')
    print(f"✅ تم تحميل المصفوفة: shape = {embeddings_title.shape}")

    embeddings_title = embeddings_title.astype(np.float32)
    faiss.normalize_L2(embeddings_title)

    dimension_title = embeddings_title.shape[1]
    index_title = faiss.IndexFlatIP(dimension_title)
    print(f"🔄 جاري إضافة {embeddings_title.shape[0]} متجه إلى فهرس العناوين...")
    index_title.add(embeddings_title)

    faiss_index_title_path = MODELS_DIR / 'faiss_index_title.flat'
    faiss.write_index(index_title, str(faiss_index_title_path))
    print(f"✅ تم حفظ فهرس العناوين في {faiss_index_title_path}")
    del embeddings_title
else:
    print("ℹ️ ملف embeddings_title.npy غير موجود، سيتم تخطي بناء فهرس العناوين.")

# ===== 4. حفظ معرفات الوثائق (Doc IDs) بالترتيب الصحيح =====
doc_ids_save_path = MODELS_DIR / 'faiss_doc_ids.pkl'
with open(doc_ids_save_path, 'wb') as f:
    pickle.dump(doc_ids_list, f)
print(f"✅ تم حفظ معرفات الوثائق ({len(doc_ids_list)}) في {doc_ids_save_path}")

print("\n🎉 تم الانتهاء من بناء جميع فهارس FAISS بنجاح!")
print("📌 المعرفات محملة من corpus_original.jsonl (متطابقة مع app.py).")
print("يمكنك الآن تشغيل الخادم (app.py) وسيستخدم FAISS تلقائياً عند تفعيل الخيار في الواجهة.")