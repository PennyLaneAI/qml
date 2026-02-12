"""
Load and cache intersphinx objects.inv files for PennyLane and Catalyst.

Tries to load from disk first; if missing or unusable, fetches from the web
and persists to disk for next time.
"""
import time
from pathlib import Path
from typing import Any

import sphobjinv as soi
import requests
from requests import exceptions as requests_exceptions

# Base URLs for objects.inv (same as filter_links.py)
PENNYLANE_OBJ_INV_URL = "https://docs.pennylane.ai/en/stable/objects.inv"
CATALYST_OBJ_INV_URL = "https://docs.pennylane.ai/projects/catalyst/en/stable/objects.inv"

REQUEST_TIMEOUT = 5  # seconds
USER_AGENT = "Mozilla/5.0 (compatible; QML-Pandoc-Filter; +https://github.com/PennyLaneAI/qml)"


def _load_inventory_from_path(path: Path) -> soi.Inventory | None:
    """Load an inventory from a local file. Returns None if file missing or invalid."""
    if not path.is_file():
        return None
    try:
        return soi.Inventory(path)
    except Exception:
        return None


def _load_inventory_from_url(
    url: str, retries: int = 3, delay: float = 1.0
) -> tuple[soi.Inventory, bytes]:
    """Download objects.inv from URL and load with retry logic. Returns (inventory, raw_bytes)."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests_exceptions.RequestException:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    # sphobjinv accepts zlib-compressed bytes from objects.inv
    raw = response.content
    return soi.Inventory(raw), raw


def _make_named_inventory(inv: soi.Inventory) -> dict[str, Any]:
    """Build a name -> item dict for easy lookup (same shape as filter_links)."""
    return {item.name: item for item in inv.objects}


class IntersphinxInventories:
    """
    Load and cache PennyLane and Catalyst objects.inv inventories.

    On initialization, each inventory is loaded from disk if present in
    cache_dir; otherwise it is fetched from the web and then saved to disk.
    The parsed contents are exposed as .pennylane and .catalyst (name -> item
    dicts) and as .pennylane_inventory / .catalyst_inventory (raw sphobjinv
    Inventory objects). Base URLs for building doc links are in .pennylane_base_url
    and .catalyst_base_url.
    """

    pennylane_base_url = "https://docs.pennylane.ai/en/stable/"
    catalyst_base_url = "https://docs.pennylane.ai/projects/catalyst/en/stable/"

    def __init__(
        self,
        cache_dir: Path | str | None = None,
    ):
        if cache_dir is None:
            cache_dir = Path.cwd() / ".intersphinx_cache"
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._pennylane_inventory: soi.Inventory | None = None
        self._catalyst_inventory: soi.Inventory | None = None
        self._pennylane: dict[str, Any] = {}
        self._catalyst: dict[str, Any] = {}

        self._load_pennylane()
        self._load_catalyst()

    def _load_pennylane(self) -> None:
        cache_path = self._cache_dir / "pennylane_objects.inv"
        inv = _load_inventory_from_path(cache_path)
        if inv is None:
            inv, raw = _load_inventory_from_url(PENNYLANE_OBJ_INV_URL)
            cache_path.write_bytes(raw)
        self._pennylane_inventory = inv
        self._pennylane = _make_named_inventory(inv)

    def _load_catalyst(self) -> None:
        cache_path = self._cache_dir / "catalyst_objects.inv"
        inv = _load_inventory_from_path(cache_path)
        if inv is None:
            inv, raw = _load_inventory_from_url(CATALYST_OBJ_INV_URL)
            cache_path.write_bytes(raw)
        self._catalyst_inventory = inv
        self._catalyst = _make_named_inventory(inv)

    @property
    def pennylane(self) -> dict[str, Any]:
        """PennyLane inventory as name -> item dict for link resolution."""
        return self._pennylane

    @property
    def catalyst(self) -> dict[str, Any]:
        """Catalyst inventory as name -> item dict for link resolution."""
        return self._catalyst

    @property
    def pennylane_inventory(self) -> soi.Inventory | None:
        """Raw PennyLane sphobjinv Inventory."""
        return self._pennylane_inventory

    @property
    def catalyst_inventory(self) -> soi.Inventory | None:
        """Raw Catalyst sphobjinv Inventory."""
        return self._catalyst_inventory
