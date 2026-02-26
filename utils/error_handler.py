from functools import wraps
import logging

logger = logging.getLogger(__name__)


def handle_errors(default_return=None):
    """Decorator for centralized error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator
