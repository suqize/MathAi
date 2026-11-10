"""Load the input and illustrative output extracted from the patent example DOCX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "pgps9k_patent_example.json"


def load_patent_example() -> dict[str, Any]:
    """Load the normalized example extracted from pgps9k_patent_example.docx."""
    with EXAMPLE_PATH.open("r", encoding="utf-8") as file:
        sample = json.load(file)
    for field in ("image", "heatmap_image"):
        sample["input"][field] = str((PROJECT_ROOT / sample["input"][field]).resolve())
    return sample
