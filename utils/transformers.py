def is_accelerator(obj):
    """Helper to check if object is a valid Accelerator instance without strict type checking"""
    return hasattr(obj, 'prepare') and hasattr(obj, 'device') and hasattr(obj, 'unwrap_model')