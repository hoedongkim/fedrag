"""Shared 501 response for endpoints whose feature lands in a later phase."""

from typing import Any, NoReturn

from fastapi import HTTPException, status

from app.schemas.common import ErrorResponse

NOT_IMPLEMENTED: dict[int | str, dict[str, Any]] = {
    status.HTTP_501_NOT_IMPLEMENTED: {
        "model": ErrorResponse,
        "description": "Not implemented yet: the feature is planned for a later phase.",
    }
}


def not_implemented(phase: int, feature: str) -> NoReturn:
    """Raise the standard 501 for an endpoint that is part of the contract but not built yet."""
    raise HTTPException(
        status.HTTP_501_NOT_IMPLEMENTED,
        f"Not implemented yet - planned for Phase {phase} ({feature}).",
    )
