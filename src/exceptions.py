import logging

logger = logging.getLogger(__name__)


class RedrobPipelineError(Exception):
    """Base exception for all pipeline errors."""


class DataLoadError(RedrobPipelineError):
    """Raised when >10% of candidate records fail validation."""


class JDParseError(RedrobPipelineError):
    """Raised when JD parsing fails after retry."""


class IndexBuildError(RedrobPipelineError):
    """Raised when <90% of candidates are successfully indexed."""


class IndexNotFoundError(RedrobPipelineError):
    """Raised when the query pipeline cannot find a pre-built index."""


class ScoringError(RedrobPipelineError):
    """Raised when the shortlist produces fewer than top_n_output scored candidates."""
