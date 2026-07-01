import logging
from pathlib import Path

from src.config import Settings
from src.embedder.index_builder import build_candidate_index

logger = logging.getLogger(__name__)


def run_indexing(
    jsonl_path: Path,
    processed_dir: Path,
    settings: Settings,
) -> None:
    """Offline pipeline. Runs once (or when dataset changes)."""
    logger.info("Starting indexing pipeline...")
    build_candidate_index(jsonl_path, processed_dir, settings)
    logger.info("Indexing pipeline complete.")
