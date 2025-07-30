import re


def slugify(text: str) -> str:
    """Convert a string to a slug-friendly format."""
    return re.sub(r"[^\w\s-]", "", text).strip().lower().replace(" ", "_")
