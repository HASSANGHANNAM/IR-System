"""
Database Initializer - Handles the startup database population.
Contains only one public function: initialize_database().
"""

import logging

logger = logging.getLogger(__name__)

def initialize_database(db_manager, config):
    """Populate SQLite tables with documents, queries, and qrels."""
    logger.info("Initializing database tables...")
    db_manager.init_documents(config.DOCUMENTS_JSONL_PATH)
    db_manager.init_queries(config.QUERIES_JSONL_PATH)
    db_manager.init_qrels(config.QRELS_PATHS)
    logger.info("Database initialization completed.")