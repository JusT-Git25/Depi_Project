"""API Key authentication using constant-time comparison and environment configuration."""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_ENV_VAR = "LIVER_API_KEY"

api_key_header_scheme = APIKeyHeader(
    name=API_KEY_HEADER_NAME,
    auto_error=False,
    description="API key required for authenticated endpoints (send via `X-API-Key` header).",
)


def get_configured_api_key() -> str:
    """Read configured API key strictly from environment variables (never hardcoded)."""
    return os.getenv(API_KEY_ENV_VAR, "").strip()


async def verify_api_key(
    provided_key: str | None = Depends(api_key_header_scheme),
) -> str:
    """Validate `X-API-Key` header using constant-time comparison.

    Returns HTTP 401 when credentials are missing or invalid, and fails closed
    (HTTP 503) if the server has not configured `LIVER_API_KEY`.
    """
    expected_key = get_configured_api_key()
    if not expected_key:
        logger.error("Server misconfiguration: environment variable %s is not set.", API_KEY_ENV_VAR)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured on the server.",
        )

    if not provided_key or not hmac.compare_digest(provided_key.strip(), expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid key in the 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return provided_key
