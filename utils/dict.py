
def prepend_key_to_dict(key_prefix: str, d: dict) -> dict:
    """Prepends a key prefix to all keys in a dictionary.

    Args:
        key_prefix (str): The prefix to prepend.
        d (dict): The original dictionary.

    Returns:
        dict: A new dictionary with the key prefix prepended to all keys.
    """
    return {f"{key_prefix}{k}": v for k, v in d.items()}