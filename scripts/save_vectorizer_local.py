import pickle
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path('data/processed_stemming')
MODELS_DIR = Path('models')

def identity(x):
    return x

def split_tokens(x):
    return x.split()

print("🔄 1. جاري تحميل النصوص المنظفة من جهازك (382,545 وثيقة)...")
titles = []
with open(DATA_DIR / 'corpus_cleaned_text.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        titles.append(' '.join(data['cleaned_tokens']))

print(f"✅ تم تحميل {len(titles)} وثيقة.")

print("🔄 2. جاري تدريب الـ TfidfVectorizer (قد يستغرق 10-15 دقيقة)...")
vectorizer = TfidfVectorizer(
    tokenizer=split_tokens,
    preprocessor=identity,
    lowercase=False,
    token_pattern=None,
    max_features=50000,
    norm='l2',
    use_idf=True
)
vectorizer.fit(titles)

print("🔄 3. جاري حفظ النموذج في مجلد models/...")
with open(MODELS_DIR / 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ تم الحفظ! الآن اشتغل على تعديل ملف app.py واشتغل السيرفر.")
