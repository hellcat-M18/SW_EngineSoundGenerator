"""Custom exceptions."""
from __future__ import annotations


class EngineToolError(Exception):
    """Base exception for the engine tool."""


class LoopProcessingError(EngineToolError):
    """Raised when loop generation fails."""


class CompilerError(EngineToolError):
    """Raised when the component compiler fails."""
