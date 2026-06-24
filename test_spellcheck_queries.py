#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
سكربت تجريبي لاختبار التصحيح الإملائي (Spell Checking) على استعلامات بها أخطاء.
يستورد دالة refine_query_text من app.py ويطبقها على قائمة من الاستعلامات الخاطئة،
ثم يعرض النتائج في جدول بسيط.
"""

# ============================================================
# تعريف الدوال المفقودة للـ pickle (لتجنب AttributeError عند تحميل app.py)
# ============================================================
def identity(x):
    return x

def split_tokens(x):
    return x.split()

# ============================================================
# استيراد الدالة من app.py
# ============================================================
from app import refine_query_text

# ============================================================
# قائمة الاستعلامات الخاطئة (المُدخلة)
# ============================================================
MISSPELLED_QUERIES = [
    "shood teechers git tenur?",
    "is veiping with e-sigarets safe?",
    "shood insidder trading bee alowed?",
    "shood corporel punisment be used in skools?",
    "shood social secutiry bee privatized?",
    "is a colledge edcation worth it?",
    "shood felons who hav compleeted ther sentence be alowed to vote?",
    "shood aborshun be legal?",
    "shood studints have to ware skool uniforms?",
    "shood any vacines be requiered for childern?",
    "shood perfomance-enhancing drugs be aksepted in sports?",
    "shood birth control pills be availabel over the counter?",
    "can alternativ energy effectivly replace fosil fuels?",
    "is sexual orientashun determined at birth?",
    "shood animels be used for scientifik or comercial testing?",
    "shood prescripshun drugs be advertised directli to consumers?",
    "shood recreational marihuana be legal?",
    "shood churces remain tax-exempt?",
    "is drinking milk helthy for humans?",
    "shood the federal minimum wage be increesed?"
]

# ============================================================
# الدالة الرئيسية
# ============================================================
def run_test():
    print("="*80)
    print("🧪 اختبار التصحيح الإملائي (Spell Checking)")
    print("="*80)
    print(f"عدد الاستعلامات: {len(MISSPELLED_QUERIES)}")
    print("\nالنتائج:\n")
    
    # طباعة رأس الجدول
    print(f"{'#':<3} {'الاستعلام الخاطئ':<40} {'الاستعلام المُصحح':<40}")
    print("-"*90)
    
    for idx, q in enumerate(MISSPELLED_QUERIES, 1):
        result = refine_query_text(q)
        corrected = result['expanded']
        # اختصار النصوص إذا كانت طويلة
        if len(q) > 37:
            q_short = q[:34] + "..."
        else:
            q_short = q
        if len(corrected) > 37:
            c_short = corrected[:34] + "..."
        else:
            c_short = corrected
        
        print(f"{idx:<3} {q_short:<40} {c_short:<40}")

    print("\n" + "="*80)
    print("✅ تم الاختبار. يمكنك مقارنة الاستعلامات الأصلية والمصححة.")
    print("   شغّل السيرفر واختبر يدوياً على الواجهة لمشاهدة تأثير التصحيح على النتائج.")

# ============================================================
# تشغيل السكربت
# ============================================================
if __name__ == "__main__":
    run_test()