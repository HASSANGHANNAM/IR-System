# core/models.py
"""
نماذج البيانات الأساسية (DTOs)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from datetime import datetime

@dataclass
class Document:
    """تمثل وثيقة في النظام"""
    doc_id: int
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """نتيجة بحث فردية"""
    doc_id: int
    score: float
    model: str  # bm25, dense_text, dense_title, hybrid, tfidf
    document: Optional[Document] = None
    
    def to_dict(self):
        return {
            'doc_id': self.doc_id,
            'score': self.score,
            'model': self.model,
            'document': self.document.to_dict() if self.document else None
        }

@dataclass
class SearchQuery:
    """استعلام بحث"""
    query: str
    top_k: int = 10
    model: str = 'hybrid'  # bm25, dense, hybrid, tfidf
    weights: Optional[Dict[str, float]] = None

@dataclass
class BaseResponse:
    """استجابة موحدة لجميع APIs"""
    status: str  # success, error, warning
    message: str
    data: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            'status': self.status,
            'message': self.message,
            'data': self.data,
            'timestamp': self.timestamp
        }