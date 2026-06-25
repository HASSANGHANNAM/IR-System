"""
سكربت بناء فهارس FAISS من ملفات Embeddings الموجودة.
يُشغل مرة واحدة فقط لإنشاء الملفات التالية في مجلد models/:
- faiss_index_text.flat
- faiss_index_title.flat (اختياري)
- faiss_doc_ids.pkl
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import faiss  # تأكد من تثبيت المكتبة: pip install faiss-cpu

# ===== إعداد المسارات =====
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'

# التأكد من وجود مجلد models
MODELS_DIR.mkdir(exist_ok=True)

# ===== 1. تحميل معرفات الوثائق (Doc IDs) =====
# نحاول تحميلها من ملف doc_ids.pkl أو من ملف JSON
doc_ids_path = MODELS_DIR / 'doc_ids.pkl'
doc_ids_list = []

if doc_ids_path.exists():
    try:
        with open(doc_ids_path, 'rb') as f:
            doc_ids_list = pickle.load(f)
        print(f"✅ تم تحميل {len(doc_ids_list)} معرف وثيقة من {doc_ids_path}")
    except Exception as e:
        print(f"⚠️ فشل تحميل doc_ids.pkl: {e}")

# إذا لم نجد doc_ids.pkl، نحاول إنشاء قائمة افتراضية بناءً على حجم المصفوفة
if not doc_ids_list:
    print("⚠️ لم يتم العثور على doc_ids.pkl. سيتم إنشاء معرفات رقمية افتراضية.")
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

# ===== 4. حفظ معرفات الوثائق (Doc IDs) =====
# إذا لم تكن doc_ids_list محملة، ننشئها بناءً على حجم الفهرس
if not doc_ids_list:
    # نحاول معرفة العدد من الفهرس الذي بنيناه
    # نقرأ عدد المتجهات من الفهرس (نفس عدد doc_ids)
    num_vectors = index_text.ntotal
    doc_ids_list = [str(i) for i in range(num_vectors)]
    print(f"⚠️ تم إنشاء {num_vectors} معرف وثيقة افتراضي (أرقام تسلسلية).")

# حفظ doc_ids_list
doc_ids_save_path = MODELS_DIR / 'faiss_doc_ids.pkl'
with open(doc_ids_save_path, 'wb') as f:
    pickle.dump(doc_ids_list, f)
print(f"✅ تم حفظ معرفات الوثائق ({len(doc_ids_list)}) في {doc_ids_save_path}")

print("\n🎉 تم الانتهاء من بناء جميع فهارس FAISS بنجاح!")
print("يمكنك الآن تشغيل الخادم (app.py) وسيستخدم FAISS تلقائياً عند تفعيل الخيار في الواجهة.")