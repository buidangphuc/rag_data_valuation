import hashlib

def calculate_chunk_hash(text: str) -> str:
    """Calculates a stable MD5 hash for a given text chunk."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()
