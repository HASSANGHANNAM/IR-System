"""
Application Entry Point - Responsible for initializing the database and running the server.
Contains only one public function: main().
"""

import logging
import sys

from app_factory import create_app
from infrastructure.db_initializer import initialize_database
import config


# ✅ فلتر لمنع رسائل طلبات HTTP من الظهور في الطرفية
class HttpRequestFilter(logging.Filter):
    """Filter out log messages that contain HTTP request methods (GET, POST, etc.)."""
    def filter(self, record):
        msg = record.getMessage()
        # إذا احتوت الرسالة على أي من هذه الكلمات، نمنع ظهورها في الطرفية
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']
        return not any(method in msg for method in methods)


def main():
    """Initialize the database, build the app, and start the server."""
    
    # ✅ إنشاء مجلد logs إذا لم يكن موجوداً
    config.LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. إعداد FileHandler المشترك =====
    # يكتب في الملف ويمسح محتواه عند كل بداية تشغيل (mode='w')
    file_handler = logging.FileHandler(config.LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
    
    # ===== 2. إعداد مسجل الجذر (Root Logger) للتطبيق =====
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # إزالة أي handlers موجودة مسبقاً (لتجنب التكرار)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # ✅ إضافة FileHandler فقط → رسائل التطبيق تذهب إلى الملف فقط (ليس الطرفية)
    root_logger.addHandler(file_handler)
    
    # ===== 3. إعداد مسجل Werkzeug (خادم Flask) =====
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO)
    
    # إزالة أي handlers موجودة مسبقاً
    for handler in werkzeug_logger.handlers[:]:
        werkzeug_logger.removeHandler(handler)
    
    # ===== 3a. StreamHandler → الطرفية (مع فلتر لمنع GET/POST) =====
    # يسمح فقط برسائل بدء التشغيل (WARNING, Running on, Press CTRL+C)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    stream_handler.addFilter(HttpRequestFilter())  # ❌ يمنع GET/POST من الطرفية
    werkzeug_logger.addHandler(stream_handler)
    
    # ===== 3b. FileHandler → ملف اللوغ (يسجل كل شيء، بما في ذلك GET/POST) =====
    # ✅ يسمح بكتابة طلبات HTTP في الملف
    werkzeug_logger.addHandler(file_handler)
    
    # منع Werkzeug من التوريث إلى الجذر (لتجنب الكتابة المزدوجة في الملف)
    werkzeug_logger.propagate = False
    
    logger = logging.getLogger(__name__)
    
    # ✅ بناء التطبيق وتهيئة قاعدة البيانات
    app, db_manager = create_app()
    initialize_database(db_manager, config)
    
    logger.info("All components ready. Starting the server...")
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()