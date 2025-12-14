"""Structured error types for Okonomi with user-friendly messages."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class ErrorCategory(Enum):
    """Categories for error classification."""

    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    SESSION_ERROR = "session_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "authentication_error"


@dataclass
class OkonomiError(Exception):
    """Base error class with user-friendly messaging."""

    message: str
    category: ErrorCategory
    user_message: str
    details: Optional[dict[str, Any]] = None
    recoverable: bool = True

    def __str__(self) -> str:
        return self.user_message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, category={self.category})"


@dataclass
class APIError(OkonomiError):
    """Errors from external API calls (fal, HuggingFace, etc.)."""

    provider: str = ""
    status_code: Optional[int] = None

    @classmethod
    def from_fal_error(cls, error: Exception, operation: str) -> "APIError":
        """Create APIError from fal client error."""
        error_str = str(error).lower()

        # Detect rate limiting
        if "rate" in error_str or "limit" in error_str or "429" in error_str:
            return cls(
                message=str(error),
                category=ErrorCategory.RATE_LIMIT_ERROR,
                user_message="Image generation service is busy. Retrying in a moment...",
                provider="fal",
                details={"operation": operation, "original_error": str(error)},
                recoverable=True,
            )

        # Detect authentication issues
        if "auth" in error_str or "401" in error_str or "403" in error_str:
            return cls(
                message=str(error),
                category=ErrorCategory.AUTHENTICATION_ERROR,
                user_message="Authentication failed. Please check your FAL_KEY.",
                provider="fal",
                details={"operation": operation, "original_error": str(error)},
                recoverable=False,
            )

        # Generic API error
        return cls(
            message=str(error),
            category=ErrorCategory.API_ERROR,
            user_message="Image generation service temporarily unavailable. Retrying...",
            provider="fal",
            details={"operation": operation, "original_error": str(error)},
            recoverable=True,
        )

    @classmethod
    def from_hf_error(cls, error: Exception, operation: str) -> "APIError":
        """Create APIError from HuggingFace client error."""
        error_str = str(error).lower()

        # Detect rate limiting
        if "rate" in error_str or "limit" in error_str or "429" in error_str:
            return cls(
                message=str(error),
                category=ErrorCategory.RATE_LIMIT_ERROR,
                user_message="AI model service is busy. Retrying in a moment...",
                provider="huggingface",
                details={"operation": operation, "original_error": str(error)},
                recoverable=True,
            )

        # Detect model not found
        if "not found" in error_str or "404" in error_str:
            return cls(
                message=str(error),
                category=ErrorCategory.API_ERROR,
                user_message="The requested model is not available.",
                provider="huggingface",
                details={"operation": operation, "original_error": str(error)},
                recoverable=False,
            )

        # Detect authentication issues
        if "auth" in error_str or "401" in error_str or "403" in error_str:
            return cls(
                message=str(error),
                category=ErrorCategory.AUTHENTICATION_ERROR,
                user_message="Authentication failed. Please check your HF_TOKEN.",
                provider="huggingface",
                details={"operation": operation, "original_error": str(error)},
                recoverable=False,
            )

        # Generic API error
        return cls(
            message=str(error),
            category=ErrorCategory.API_ERROR,
            user_message="AI model service temporarily unavailable. Retrying...",
            provider="huggingface",
            details={"operation": operation, "original_error": str(error)},
            recoverable=True,
        )


@dataclass
class TimeoutError(OkonomiError):
    """Timeout errors with context."""

    timeout_seconds: float = 0.0
    operation: str = ""

    @classmethod
    def create(cls, operation: str, timeout: float) -> "TimeoutError":
        """Create a TimeoutError with standard messaging."""
        return cls(
            message=f"Operation '{operation}' timed out after {timeout}s",
            category=ErrorCategory.TIMEOUT_ERROR,
            user_message=f"The operation is taking longer than expected ({timeout}s). You can try again.",
            timeout_seconds=timeout,
            operation=operation,
            recoverable=True,
        )


@dataclass
class ValidationError(OkonomiError):
    """Input validation errors."""

    field: str = ""
    constraint: str = ""
    value: str = ""

    @classmethod
    def create(cls, field: str, constraint: str, value: str = "") -> "ValidationError":
        """Create a ValidationError with standard messaging."""
        return cls(
            message=f"Validation failed for '{field}': {constraint}",
            category=ErrorCategory.VALIDATION_ERROR,
            user_message=f"Invalid input for '{field}': {constraint}",
            field=field,
            constraint=constraint,
            value=value,
            recoverable=True,
        )

    @classmethod
    def empty_field(cls, field: str) -> "ValidationError":
        """Create error for empty required field."""
        return cls.create(field, "cannot be empty")

    @classmethod
    def too_long(cls, field: str, max_length: int, actual_length: int) -> "ValidationError":
        """Create error for field exceeding max length."""
        return cls.create(
            field,
            f"exceeds maximum length of {max_length} characters (got {actual_length})",
        )

    @classmethod
    def invalid_format(cls, field: str, expected_format: str) -> "ValidationError":
        """Create error for invalid format."""
        return cls.create(field, f"invalid format, expected {expected_format}")


@dataclass
class ExecutionError(OkonomiError):
    """Code execution errors."""

    code_snippet: str = ""
    line_number: Optional[int] = None

    @classmethod
    def create(cls, error: Exception, code: str = "", line_number: Optional[int] = None) -> "ExecutionError":
        """Create an ExecutionError from an exception."""
        return cls(
            message=str(error),
            category=ErrorCategory.EXECUTION_ERROR,
            user_message=f"Code execution failed: {str(error)}",
            code_snippet=code[:500] if code else "",
            line_number=line_number,
            recoverable=True,
        )

    @classmethod
    def restricted_operation(cls, operation: str) -> "ExecutionError":
        """Create error for restricted operation attempt."""
        return cls(
            message=f"Restricted operation attempted: {operation}",
            category=ErrorCategory.EXECUTION_ERROR,
            user_message=f"The operation '{operation}' is not allowed for security reasons.",
            recoverable=False,
        )


@dataclass
class SessionError(OkonomiError):
    """Session management errors."""

    session_id: str = ""

    @classmethod
    def not_found(cls, session_id: str) -> "SessionError":
        """Create error for session not found."""
        return cls(
            message=f"Session '{session_id}' not found",
            category=ErrorCategory.SESSION_ERROR,
            user_message=f"Session '{session_id}' was not found. Starting a new session.",
            session_id=session_id,
            recoverable=True,
        )

    @classmethod
    def index_corrupted(cls) -> "SessionError":
        """Create error for corrupted session index."""
        return cls(
            message="Session index file is corrupted",
            category=ErrorCategory.SESSION_ERROR,
            user_message="Session data appears to be corrupted. Attempting recovery...",
            recoverable=True,
        )

    @classmethod
    def save_failed(cls, session_id: str, reason: str) -> "SessionError":
        """Create error for failed session save."""
        return cls(
            message=f"Failed to save session '{session_id}': {reason}",
            category=ErrorCategory.SESSION_ERROR,
            user_message=f"Could not save session: {reason}",
            session_id=session_id,
            recoverable=True,
        )
