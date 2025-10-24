import tempfile
from typing import Tuple


def save_temp_file(bytes_obj: bytes, suffix: str = ".wav") -> str:
    """Save bytes to a temporary file and return path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with open(path, "wb") as f:
        f.write(bytes_obj)
    return path

    