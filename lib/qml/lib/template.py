from typing import Any
from datetime import timezone, datetime
import inspect


def metadata(
    title: str,
    description: str,
    authors: list[str],
    thumbnail: str | None,
    large_thumbnail: str | None,
    categories: list[str],
) -> dict[str, Any]:
    today = datetime.now(tz=timezone.utc)
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)

    metadata = {
        "title": title,
        "authors": authors,
        "dateOfPublication": today.isoformat(),
        "dateOfLastModification": today.isoformat(),
        "categores": categories,
        "tags": [],
        "previewImages": [],
        "seoDescription": description,
        "doi": "",
        "references": [],
        "basedOnPapers": [],
        "referencedByPapers": [],
        "relatedContent": [],
    }

    if thumbnail:
        metadata["previewImages"].append({"type": "thumbnail", "uri": thumbnail})
    if large_thumbnail:
        metadata["previewImages"].append(
            {"type": "large_thumbnail", "uri": large_thumbnail}
        )

    return metadata


def demo(title: str) -> str:
    return inspect.cleandoc(f'''
    r"""
    {title}
    {"="}

    Introduce your demo here!
    """
    
    print("Hello")

    ###############################################################################
    #
    # Add comment blocks to separate code blocks
    #

    print("World")

    ''')
