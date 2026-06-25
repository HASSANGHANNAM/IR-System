# infrastructure/file_handlers.py
"""
دوال التعامل مع الملفات (تحميل/حفظ البيانات)
"""

import json
import pickle
from pathlib import Path
import numpy as np
from scipy import sparse

def load_pickle(path):
    """تحميل ملف Pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path):
    """حفظ ملف Pickle"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_npz(path):
    """تحميل ملف NPZ (مصفوفات متفرقة)"""
    return sparse.load_npz(path)

def save_npz(matrix, path):
    """حفظ مصفوفة متفرقة بصيغة NPZ"""
    sparse.save_npz(path, matrix)

def load_npy(path):
    """تحميل ملف NPY (مصفوفات كثيفة)"""
    return np.load(path)

def save_npy(data, path):
    """حفظ مصفوفة كثيفة بصيغة NPY"""
    np.save(path, data)

def load_jsonl(path):
    """تحميل ملف JSONL إلى قائمة"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def load_json(path):
    """تحميل ملف JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    """حفظ ملف JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)