import string
import unicodedata
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

try:
    from spellchecker import SpellChecker
    SPELL = SpellChecker()
    SPELL_AVAILABLE = True
except ImportError:
    SPELL = None
    SPELL_AVAILABLE = False

def identity(x):
    return x

def split_tokens(x):
    return x.split()

def clean_query(text: str) -> str:
    if not text or not text.strip():
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [STEMMER.stem(token) for token in tokens]
    return ' '.join(tokens)

def refine_query_text(query: str) -> dict:
    if not query or not query.strip():
        return {'normalized': query, 'expanded': query}
    tokens = query.lower().split()
    if SPELL_AVAILABLE:
        corrected_tokens = []
        misspelled = SPELL.unknown(tokens)
        for token in tokens:
            if token in misspelled:
                corrected = SPELL.correction(token)
                if corrected is not None and corrected != token:
                    corrected_tokens.append(corrected)
                else:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
    else:
        corrected_tokens = tokens
    seen = set()
    final_tokens = []
    for token in corrected_tokens:
        if token not in seen:
            seen.add(token)
            final_tokens.append(token)
    return {
        'normalized': ' '.join(corrected_tokens),
        'expanded': ' '.join(final_tokens)
    }

def normalize_scores(scores):
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)