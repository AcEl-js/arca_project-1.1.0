
import hashlib
from datetime import date

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def regulation_id_for(text: str, date_of_law: str | None) -> str:
    """Generate deterministic unique regulation ID."""
    base = (date_of_law or "") + "|" + text
    return hashlib.sha256(base.encode()).hexdigest()


def today_iso() -> str:
    return date.today().isoformat()
