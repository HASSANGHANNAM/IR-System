"""
Application Factory - Responsible for assembling the Flask application.
Contains only one public function: create_app().
"""

import logging
from flask import Flask, send_from_directory

import config
from infrastructure.database import DatabaseManager
from infrastructure.model_loader import ModelManager
from infrastructure.cache_manager import CacheManager
from services.retrieval_service import RetrievalService
from services.ranking_service import RankingService
from routes.search_routes import init_search_routes
from routes.admin_routes import init_admin_routes
from routes.chart_routes import init_chart_routes


def create_app():
    """
    Assemble and return the Flask application instance with all dependencies.
    This function does NOT run the server or initialize the database.
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Assembling application components...")
    
    # Initialize managers
    db_manager = DatabaseManager(config.DB_PATH)
    model_manager = ModelManager(config)  # Loads all models immediately
    cache_manager = CacheManager()
    
    # Initialize services
    retrieval_service = RetrievalService(config, model_manager, db_manager, cache_manager)
    ranking_service = RankingService(config, db_manager)
    
    # Create Flask app
    app = Flask(__name__, static_folder=str(config.UI_DIR), static_url_path='')
    
    # Register Blueprints
    app.register_blueprint(init_search_routes(retrieval_service, ranking_service, db_manager, cache_manager))
    app.register_blueprint(init_admin_routes(model_manager, cache_manager))
    app.register_blueprint(init_chart_routes(config.DATA_DIR, config.ROOT))
    
    # Home route
    @app.route('/')
    def home():
        return send_from_directory(app.static_folder, 'index.html')
    
    logger.info("All components assembled successfully.")
    
    return app, db_manager