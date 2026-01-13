"""Custom exceptions for total-recall."""

from typing import Any, Optional


class TotalRecallError(Exception):
    """Custom exception for total-recall errors with structured info."""

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": str(self),
            "error_code": self.error_code,
            "details": self.details,
        }
