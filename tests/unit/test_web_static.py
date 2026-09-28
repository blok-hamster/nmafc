"""Tests for serving the exported Next.js dashboard from FastAPI.

The one-port Docker story (Phase 3.4) is: `nmafc-web` serves both the /api
surface and the statically exported dashboard, with SPA fallback, file-type
allowlisting, and containment that blocks path traversal. These tests drive
_dispatch_static directly against a synthetic export tree.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from nmafc.web import app as web_app


@pytest.fixture
def export_dir(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("home")
    (tmp_path / "memory.html").write_text("memory page")
    (tmp_path / "graph").mkdir()
    (tmp_path / "graph" / "index.html").write_text("graph page")
    (tmp_path / "app.css").write_text("body{}")
    (tmp_path / "_next" / "static").mkdir(parents=True)
    (tmp_path / "_next" / "static" / "chunk.js").write_text("run()")
    monkeypatch.setenv("NMAFC_STATIC_UI_DIR", str(tmp_path))
    return tmp_path


def _serve(full_path: str):
    return web_app._dispatch_static(full_path)


def test_serves_index_at_root(export_dir):
    resp = _serve("")
    assert isinstance(resp, FileResponse)
    assert resp.path == export_dir / "index.html"


def test_serves_route_html(export_dir):
    resp = _serve("memory")
    assert resp.path == export_dir / "memory.html"


def test_serves_route_directory(export_dir):
    resp = _serve("graph")
    assert resp.path == export_dir / "graph" / "index.html"


def test_serves_nested_asset(export_dir):
    resp = _serve("_next/static/chunk.js")
    assert resp.path == export_dir / "_next" / "static" / "chunk.js"


def test_rejects_unknown_suffix(export_dir):
    with pytest.raises(HTTPException) as exc:
        _serve("store.lancedb")
    assert exc.value.status_code == 404


def test_rejects_traversal(export_dir, tmp_path):
    outside = tmp_path.parent / "escape.html"
    outside.write_text("secret")
    with pytest.raises(HTTPException) as exc:
        _serve("../escape.html")
    assert exc.value.status_code == 404


def test_missing_asset_falls_back_to_index(export_dir):
    resp = _serve("no-such-route")
    assert resp.path == export_dir / "index.html"
