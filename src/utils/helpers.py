"""General helper utilities."""

def ensure_directory(path):
    path.mkdir(parents=True, exist_ok=True)
