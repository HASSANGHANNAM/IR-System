"""
سكربت تحويل مصفوفة TF-IDF من float64/float32 إلى float16
لتوفير 50% من استهلاك الذاكرة ووقت التحميل.
يُشغل مرة واحدة فقط.
"""

import numpy as np
from scipy import sparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'

# ===== 1. التحقق من وجود الملف =====
input_path = MODELS_DIR / 'tfidf_text_matrix.npz'
if not input_path.exists():
    print(f"❌ الملف {input_path} غير موجود!")
    print("تأكد من أنك في مجلد المشروع الصحيح.")
    exit(1)

print(f"📂 جاري تحميل المصفوفة من {input_path}...")

# ===== 2. تحميل المصفوفة =====
matrix = sparse.load_npz(input_path)
print(f"✅ تم التحميل: shape = {matrix.shape}, dtype = {matrix.dtype}")
print(f"   حجم الملف الحالي: {input_path.stat().st_size / (1024**2):.2f} MB")

# ===== 3. إنشاء نسخة احتياطية (مرة واحدة فقط) =====
backup_path = MODELS_DIR / 'tfidf_text_matrix_backup.npz'
if not backup_path.exists():
    print(f"📂 جاري إنشاء نسخة احتياطية في {backup_path}...")
    sparse.save_npz(backup_path, matrix)
    print("✅ تم إنشاء النسخة الاحتياطية.")
else:
    print("ℹ️ النسخة الاحتياطية موجودة بالفعل، يتم تخطي هذه الخطوة.")

# ===== 4. تحويل البيانات إلى float16 =====
print("🔄 جاري تحويل البيانات إلى float16...")
matrix.data = matrix.data.astype(np.float16)

# ===== 5. حفظ المصفوفة الجديدة (نفس الاسم) =====
print(f"💾 جاري حفظ المصفوفة الجديدة في {input_path}...")
sparse.save_npz(input_path, matrix)
print("✅ تم حفظ المصفوفة بنجاح!")

# ===== 6. عرض تقرير المقارنة =====
new_size = input_path.stat().st_size / (1024**2)
old_size = backup_path.stat().st_size / (1024**2) if backup_path.exists() else new_size

print("\n📊 مقارنة حجم الملفات:")
print(f"   الملف القديم (احتياطي): {old_size:.2f} MB")
print(f"   الملف الجديد (float16) : {new_size:.2f} MB")
print(f"   🎉 تم توفير: {(old_size - new_size):.2f} MB ({(1 - new_size/old_size) * 100:.1f}%)")

print("\n✅ تم الانتهاء! يمكنك الآن تشغيل الخادم (app.py) كالمعتاد.")
print("ملاحظة: العمليات الحسابية ستتم بـ float32 (تحويل مؤقت) لكن التحميل والذاكرة الخاملة ستوفر 50%.")