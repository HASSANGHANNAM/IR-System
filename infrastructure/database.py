"""
Database Manager - Handles SQLite database operations.
Contains one class: DatabaseManager.
"""

import json
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    text TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    query_id TEXT PRIMARY KEY,
                    text TEXT,
                    description TEXT,
                    narrative TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS qrels (
                    query_id TEXT,
                    doc_id TEXT,
                    score REAL,
                    PRIMARY KEY (query_id, doc_id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_qrels_query_id ON qrels(query_id)')
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_documents(self, jsonl_path: Path):
        if not jsonl_path.exists():
            logger.warning(f"Documents file not found: {jsonl_path}")
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            existing = cursor.fetchone()[0]
            
            if existing > 0:
                logger.info(f"Documents table ready with {existing} documents")
                return
            
            batch = []
            inserted = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    
                    doc_id = str(data.get('doc_id', ''))
                    if not doc_id:
                        continue
                    
                    title = data.get('title', '')
                    text = data.get('text', '')
                    batch.append((doc_id, title, text))
                    
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO documents (doc_id, title, text) VALUES (?, ?, ?)',
                            batch
                        )
                        conn.commit()
                        inserted += len(batch)
                        batch.clear()
            
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO documents (doc_id, title, text) VALUES (?, ?, ?)',
                    batch
                )
                conn.commit()
                inserted += len(batch)
            
            logger.info(f"Stored {inserted} documents in SQLite")
    
    def init_queries(self, jsonl_path: Path):
        if not jsonl_path.exists():
            logger.warning(f"Queries file not found: {jsonl_path}")
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM queries')
            existing = cursor.fetchone()[0]
            
            if existing > 0:
                logger.info(f"Queries table ready with {existing} queries")
                return
            
            batch = []
            inserted = 0
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    
                    query_id = str(data.get('query_id', ''))
                    if not query_id:
                        continue
                    
                    text = data.get('text') or data.get('original_text') or data.get('query') or ''
                    description = data.get('description', '')
                    narrative = data.get('narrative', '')
                    batch.append((query_id, text, description, narrative))
                    
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO queries (query_id, text, description, narrative) VALUES (?, ?, ?, ?)',
                            batch
                        )
                        conn.commit()
                        inserted += len(batch)
                        batch.clear()
            
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO queries (query_id, text, description, narrative) VALUES (?, ?, ?, ?)',
                    batch
                )
                conn.commit()
                inserted += len(batch)
            
            logger.info(f"Stored {inserted} queries in SQLite")
    
    def init_qrels(self, qrels_paths: list):
        qrels_path = next((p for p in qrels_paths if p.exists()), None)
        if qrels_path is None:
            logger.warning("qrels file not found")
            return
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM qrels')
            existing = cursor.fetchone()[0]
            
            if existing > 0:
                logger.info(f"qrels table ready with {existing} rows")
                return
            
            batch = []
            inserted = 0
            with open(qrels_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('query-id'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) < 3:
                        parts = line.split()
                    if len(parts) < 3:
                        continue
                    
                    query_id, doc_id, score = parts[0], parts[1], parts[2]
                    try:
                        score_value = float(score)
                    except ValueError:
                        continue
                    
                    batch.append((str(query_id), str(doc_id), score_value))
                    
                    if len(batch) >= 5000:
                        cursor.executemany(
                            'INSERT OR REPLACE INTO qrels (query_id, doc_id, score) VALUES (?, ?, ?)',
                            batch
                        )
                        conn.commit()
                        inserted += len(batch)
                        batch.clear()
            
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO qrels (query_id, doc_id, score) VALUES (?, ?, ?)',
                    batch
                )
                conn.commit()
                inserted += len(batch)
            
            logger.info(f"Stored {inserted} qrels rows from {qrels_path}")
    
    def get_documents_texts(self, doc_ids: list) -> dict:
        if not doc_ids:
            return {}
        
        placeholders = ','.join(['?'] * len(doc_ids))
        query = f'SELECT doc_id, text FROM documents WHERE doc_id IN ({placeholders})'
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, [str(doc_id) for doc_id in doc_ids])
            return {str(doc_id): text for doc_id, text in cursor.fetchall()}
    
    def get_document_titles(self, doc_ids: list) -> dict:
        if not doc_ids:
            return {}
        
        placeholders = ','.join(['?'] * len(doc_ids))
        query = f'SELECT doc_id, title FROM documents WHERE doc_id IN ({placeholders})'
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, [str(doc_id) for doc_id in doc_ids])
            return {str(doc_id): title for doc_id, title in cursor.fetchall()}
    
    def get_document_by_id(self, doc_id: str) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_id, title, text FROM documents WHERE doc_id = ?", (str(doc_id),))
            row = cursor.fetchone()
            if row:
                return {'doc_id': row[0], 'title': row[1], 'text': row[2]}
            return None
    
    def get_queries(self) -> list:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT query_id, text, description, narrative FROM queries ORDER BY CAST(query_id AS INTEGER), query_id')
            rows = cursor.fetchall()
            return [
                {
                    'query_id': row['query_id'],
                    'text': row['text'],
                    'description': row['description'],
                    'narrative': row['narrative'],
                }
                for row in rows
            ]
    
    def get_query_text_by_id(self, query_id: str) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT text FROM queries WHERE query_id = ?', (str(query_id),))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_qrels_for_query(self, query_id: str) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT doc_id, score FROM qrels WHERE query_id = ? AND score > 0 ORDER BY score DESC, doc_id',
                (str(query_id),)
            )
            return [{'doc_id': str(doc_id), 'score': float(score)} for doc_id, score in cursor.fetchall()]