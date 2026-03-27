"""Integration tests for create_diagram, add_node, and connect_nodes.

Requires:
  - Diag backend + Redis running
  - DEV_KEY env var set to a valid tld_<hex> API key
  - DIAG_BASE_URL env var (defaults to http://localhost:8080)

Run:
  DEV_KEY=tld_... uv run pytest tests/test_integration.py -v
"""

import os
import time
from unittest.mock import MagicMock

import httpx
import pytest

# Ensure REDIS_HOST defaults to localhost for local test runs
os.environ.setdefault("REDIS_HOST", "localhost")

from main import DIAG_BASE_URL, _diagram_key, _get_redis, _node_key, add_node, connect_nodes, create_diagram
from main import auth as mcp_auth

DEV_KEY = os.environ.get("DEV_KEY", "")
DIAG_USERNAME = os.environ.get("DIAG_USERNAME", "admin")
DIAG_PASSWORD = os.environ.get("DIAG_PASSWORD", "password")


def _ctx():
    """Return a minimal stub that satisfies the Context type hint."""
    ctx = MagicMock()
    ctx.client_id = None  # tools don't use ctx.client_id; they use api_key arg directly
    return ctx


def _rest_client() -> httpx.Client:
    """Return an httpx.Client authenticated as admin via session cookie.

    Used to verify that MCP operations actually persisted data to the DB.
    """
    client = httpx.Client(base_url=DIAG_BASE_URL, timeout=15)
    resp = client.post("/api/auth/login", json={"username": DIAG_USERNAME, "password": DIAG_PASSWORD})
    if resp.status_code != 200:
        raise RuntimeError(f"REST login failed ({resp.status_code}): {resp.text}")
    return client


@pytest.fixture(scope="module")
def api_key():
    if not DEV_KEY:
        pytest.skip("DEV_KEY env var not set")
    return DEV_KEY


@pytest.fixture(scope="module")
def rest_client(api_key):
    """Authenticated REST client for DB verification (module-scoped)."""
    try:
        return _rest_client()
    except RuntimeError as e:
        pytest.skip(f"Could not create REST client: {e}")


@pytest.fixture(scope="module")
def diagram_slug(api_key):
    """Create a diagram once for all tests in this module."""
    name = f"Integration Test Diagram {int(time.time())}"
    result = create_diagram(_ctx(), api_key=api_key, name=name, description="Created by integration tests")
    assert "diagram" in result
    return result["diagram"]


def _resolve_diagram_uuid(api_key: str, diagram_slug: str) -> str:
    """Return the diagram UUID from Redis cache."""
    token_data = mcp_auth.verify_token(api_key)
    assert token_data, "API key verification failed"
    org_slug = token_data["org_slug"]
    uuid = _get_redis().get(_diagram_key(org_slug, diagram_slug))
    assert uuid, f"Diagram slug '{diagram_slug}' not found in Redis"
    return str(uuid)


class TestCreateDiagram:
    def test_returns_diagram_slug(self, api_key):
        result = create_diagram(_ctx(), api_key=api_key, name=f"Test Diag {int(time.time())}")
        assert isinstance(result["diagram"], str)
        assert len(result["diagram"]) > 0

    def test_returns_name(self, api_key):
        name = f"Named Diagram {int(time.time())}"
        result = create_diagram(_ctx(), api_key=api_key, name=name)
        assert result["name"] == name

    def test_optional_fields_returned(self, api_key):
        result = create_diagram(_ctx(), api_key=api_key, name=f"Full Diagram {int(time.time())}", description="desc", level_label="Package")
        assert result["description"] == "desc"
        assert result["level_label"] == "Package"

    def test_invalid_key_raises(self):
        with pytest.raises(RuntimeError, match="Unauthorized"):
            create_diagram(_ctx(), api_key="tld_invalid", name="Should Fail")

    def test_diagram_persisted_in_db(self, api_key, rest_client):
        """Verify the diagram is actually written to the database."""
        name = f"DB Check Diagram {int(time.time())}"
        create_diagram(_ctx(), api_key=api_key, name=name)

        resp = rest_client.get("/api/diagrams")
        assert resp.status_code == 200, f"GET /api/diagrams failed: {resp.text}"
        names = [d["name"] for d in resp.json()]
        assert name in names, f"Diagram '{name}' not found in DB. Got: {names}"


class TestAddNode:
    def test_returns_node_slug(self, api_key, diagram_slug):
        result = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name="Auth Service", type="system")
        assert isinstance(result["node"], str)
        assert len(result["node"]) > 0

    def test_returns_name_and_type(self, api_key, diagram_slug):
        result = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name="Database", type="database")
        assert result["name"] == "Database"
        assert result["type"] == "database"

    def test_optional_fields_returned(self, api_key, diagram_slug):
        result = add_node(
            _ctx(), api_key=api_key, diagram=diagram_slug, name="API Gateway", type="component", description="Routes requests", technology="Go", tags=["backend", "critical"]
        )
        assert result["description"] == "Routes requests"
        assert result["technology"] == "Go"

    def test_invalid_diagram_raises(self, api_key):
        with pytest.raises(RuntimeError):
            add_node(_ctx(), api_key=api_key, diagram="nonexistent-diagram-slug", name="X", type="system")

    def test_invalid_key_raises(self, diagram_slug):
        with pytest.raises(RuntimeError, match="Unauthorized"):
            add_node(_ctx(), api_key="tld_invalid", diagram=diagram_slug, name="X", type="system")

    def test_node_persisted_in_db(self, api_key, diagram_slug, rest_client):
        """Verify the node (object + diagram placement) is actually written to the database."""
        node_name = f"DB Check Node {int(time.time())}"
        add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name=node_name, type="system")

        diagram_uuid = _resolve_diagram_uuid(api_key, diagram_slug)
        resp = rest_client.get(f"/api/diagrams/{diagram_uuid}/objects")
        assert resp.status_code == 200, f"GET /api/diagrams/{diagram_uuid}/objects failed: {resp.text}"
        obj_names = [o["name"] for o in resp.json()]
        assert node_name in obj_names, f"Node '{node_name}' not found in DB. Got: {obj_names}"


class TestConnectNodes:
    @pytest.fixture(scope="class")
    def two_nodes(self, api_key, diagram_slug):
        src = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name="Source Service", type="system")
        tgt = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name="Target Service", type="system")
        return src["node"], tgt["node"]

    def test_returns_edge_details(self, api_key, diagram_slug, two_nodes):
        src_slug, tgt_slug = two_nodes
        result = connect_nodes(_ctx(), api_key=api_key, diagram=diagram_slug, source_node=src_slug, target_node=tgt_slug, label="calls")
        assert result["diagram"] == diagram_slug
        assert result["source_node"] == src_slug
        assert result["target_node"] == tgt_slug
        assert result["label"] == "calls"

    def test_direction_forward(self, api_key, diagram_slug, two_nodes):
        src_slug, tgt_slug = two_nodes
        result = connect_nodes(_ctx(), api_key=api_key, diagram=diagram_slug, source_node=src_slug, target_node=tgt_slug, direction="forward")
        assert result["direction"] == "forward"

    def test_invalid_node_raises(self, api_key, diagram_slug, two_nodes):
        src_slug, _ = two_nodes
        with pytest.raises(RuntimeError):
            connect_nodes(_ctx(), api_key=api_key, diagram=diagram_slug, source_node=src_slug, target_node="nonexistent-node-slug")

    def test_invalid_key_raises(self, diagram_slug, two_nodes):
        src_slug, tgt_slug = two_nodes
        with pytest.raises(RuntimeError, match="Unauthorized"):
            connect_nodes(_ctx(), api_key="tld_invalid", diagram=diagram_slug, source_node=src_slug, target_node=tgt_slug)

    def test_edge_persisted_in_db(self, api_key, diagram_slug, rest_client):
        """Verify the edge is actually written to the database."""
        src = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name=f"Edge Src {int(time.time())}", type="system")
        tgt = add_node(_ctx(), api_key=api_key, diagram=diagram_slug, name=f"Edge Tgt {int(time.time())}", type="system")
        label = f"db-verify-{int(time.time())}"
        connect_nodes(_ctx(), api_key=api_key, diagram=diagram_slug, source_node=src["node"], target_node=tgt["node"], label=label)

        diagram_uuid = _resolve_diagram_uuid(api_key, diagram_slug)
        resp = rest_client.get(f"/api/diagrams/{diagram_uuid}/edges")
        assert resp.status_code == 200, f"GET /api/diagrams/{diagram_uuid}/edges failed: {resp.text}"
        edge_labels = [e.get("label") for e in resp.json()]
        assert label in edge_labels, f"Edge with label '{label}' not found in DB. Got: {edge_labels}"
