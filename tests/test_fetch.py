from __future__ import annotations

import pytest

from gehrman.catalogs.fetch import CatalogFetchError, fetch_catalog


def test_rejects_non_https() -> None:
    with pytest.raises(CatalogFetchError, match="non-HTTPS"):
        fetch_catalog("http://example.com/data.json")


def test_rejects_file_scheme() -> None:
    with pytest.raises(CatalogFetchError, match="non-HTTPS"):
        fetch_catalog("file:///etc/passwd")
