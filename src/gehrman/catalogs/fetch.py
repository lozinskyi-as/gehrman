"""HTTPS download of public bug-bounty catalog JSON files.

Rules enforced here:
- HTTPS only. A non-HTTPS URL is a programming error and is refused.
- Hard size cap so a hostile mirror cannot exhaust memory.
- Fixed timeout. No retries; the caller decides.
- The body is returned as raw bytes so the persisted snapshot is byte-
  identical to what the server sent (important for the SHA-256 audit).
"""

from __future__ import annotations

import httpx

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
TIMEOUT_SECONDS = 60.0


class CatalogFetchError(RuntimeError):
    """Raised for any failure while downloading a catalog."""


def fetch_catalog(url: str, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    if not url.startswith("https://"):
        raise CatalogFetchError(f"Refusing non-HTTPS URL: {url}")
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                buffer = bytearray()
                for chunk in response.iter_bytes():
                    buffer.extend(chunk)
                    if len(buffer) > max_bytes:
                        raise CatalogFetchError(
                            f"Catalog exceeds {max_bytes} bytes; aborting"
                        )
                return bytes(buffer)
    except httpx.HTTPError as exc:
        raise CatalogFetchError(f"Download failed: {exc}") from exc
