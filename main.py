"""
Diag FastMCP Server
-------------------
Exposes the DiagramService gRPC API (via ConnectRPC JSON protocol) as MCP tools
so LLMs can programmatically create architecture diagrams, add nodes, and connect them.

Authentication (checked in order):
  1. DIAG_API_KEY env var  — a diag API key (diag_sk_<hex>)
  2. DIAG_SESSION_TOKEN env var — a session UUID from /api/auth/login
  3. DIAG_USERNAME + DIAG_PASSWORD env vars — auto-login on first call

Slug translation layer (Redis db1):
  - All tool inputs/outputs use human-readable slugs instead of UUIDs.
  - Slug → UUID mappings are cached in Redis db1.
  - If a slug is not found in the cache, it is assumed not to exist.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import httpx
import redis as redis_lib
from fastmcp import FastMCP
from pydantic import Field

# ── Server ─────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "diag",
    instructions="""
tlDiagram is a C4-inspired interactive software architecture diagramming tool.

## Available tools

- **get_org** — Retrieve your organisation's slug (required for all other tools).
- **create_diagram** — Create a new diagram canvas. Returns a diagram.
- **add_node** — Place an architectural component on a diagram. Returns a node_slug.
- **connect_nodes** — Draw a directed edge between two nodes on the same diagram.

## Typical workflow

1. Call `get_org` once to get your org's slug.
2. Call `create_diagram` with the org_slug to create a canvas and get a diagram_slug.
3. Call `add_node` multiple times to place components; collect each returned node_slug.
4. Call `connect_nodes` to define relationships between nodes.

## Node types

Common values for the `type` parameter: "class", "function", "module", "method", "database", "queue", "file",
"interface", "object", "loop", "system", "container", "repository".

## Edge directions

- "forward" (default) — arrow points from source to target
- "backward" — arrow points from target to source
- "both" — bidirectional arrow
- "none" — no arrowhead

""",
)

# ── Configuration ──────────────────────────────────────────────────────────────

DIAG_BASE_URL = os.getenv("DIAG_BASE_URL", "http://localhost:8080").rstrip("/")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

_redis: redis_lib.Redis | None = None


def _get_redis() -> redis_lib.Redis:
    global _redis
    if _redis is None:
        _redis = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, db=1, decode_responses=True)
    return _redis


# ── Slug utilities ─────────────────────────────────────────────────────────────


def slugify(name: str) -> str:
    """Convert a name to a lowercase, hyphenated slug."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "unnamed"


def _unique_slug(base: str, exists_fn) -> str:
    """Return `base` if it's free; otherwise try base-2, base-3, …"""
    candidate = base
    counter = 2
    while exists_fn(candidate):
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


# ── Redis cache helpers ────────────────────────────────────────────────────────
#
# Key schema (Redis db1):
#   diag:org:slug:{slug}                    → org_uuid
#   diag:org:uuid:{uuid}                    → org_slug
#   diag:diagram:{org_slug}:{diagram_slug}  → diagram_uuid
#   diag:node:{org_slug}:{node_slug}        → node_uuid


def _org_slug_key(slug: str) -> str:
    return f"diag:org:slug:{slug}"


def _org_uuid_key(uuid: str) -> str:
    return f"diag:org:uuid:{uuid}"


def _diagram_key(org_slug: str, diagram_slug: str) -> str:
    return f"diag:diagram:{org_slug}:{diagram_slug}"


def _node_key(org_slug: str, node_slug: str) -> str:
    return f"diag:node:{org_slug}:{node_slug}"


def cache_org(slug: str, uuid: str) -> None:
    r = _get_redis()
    r.set(_org_slug_key(slug), uuid)
    r.set(_org_uuid_key(uuid), slug)


def resolve_org_slug(slug: str) -> str:
    uuid = _get_redis().get(_org_slug_key(slug))
    if not uuid:
        raise RuntimeError(f"Organisation '{slug}' not found in cache. Call get_org() first.")
    return uuid


def cache_diagram(org_slug: str, diagram_slug: str, diagram_uuid: str) -> None:
    _get_redis().set(_diagram_key(org_slug, diagram_slug), diagram_uuid)


def resolve_diagram_slug(org_slug: str, diagram_slug: str) -> str:
    uuid = _get_redis().get(_diagram_key(org_slug, diagram_slug))
    if not uuid:
        raise RuntimeError(f"Diagram '{diagram_slug}' not found in cache for org '{org_slug}'.")
    return uuid


def cache_node(org_slug: str, node_slug: str, node_uuid: str) -> None:
    _get_redis().set(_node_key(org_slug, node_slug), node_uuid)


def resolve_node_slug(org_slug: str, node_slug: str) -> str:
    uuid = _get_redis().get(_node_key(org_slug, node_slug))
    if not uuid:
        raise RuntimeError(f"Node '{node_slug}' not found in cache for org '{org_slug}'.")
    return uuid


def _diagram_slug_exists(org_slug: str, slug: str) -> bool:
    return bool(_get_redis().exists(_diagram_key(org_slug, slug)))


def _node_slug_exists(org_slug: str, slug: str) -> bool:
    return bool(_get_redis().exists(_node_key(org_slug, slug)))


# ── Auth ───────────────────────────────────────────────────────────────────────

_session_token: str | None = None


def _get_auth_header() -> str:
    """Return the Authorization header value, obtaining a session if needed."""
    global _session_token

    api_key = os.getenv("DIAG_API_KEY", "").strip()
    if api_key:
        return f"Bearer {api_key}"

    session = os.getenv("DIAG_SESSION_TOKEN", "").strip()
    if session:
        return f"Bearer {session}"

    if _session_token:
        return f"Bearer {_session_token}"

    username = os.getenv("DIAG_USERNAME", "").strip()
    password = os.getenv("DIAG_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError("No authentication configured. Set DIAG_API_KEY, DIAG_SESSION_TOKEN, or DIAG_USERNAME + DIAG_PASSWORD environment variables.")

    resp = httpx.post(f"{DIAG_BASE_URL}/api/auth/login", json={"username": username, "password": password}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text}")
    data = resp.json()
    _session_token = data.get("session_id") or data.get("sessionId") or data.get("token")
    if not _session_token:
        raise RuntimeError(f"Login succeeded but no session token in response: {data}")
    return f"Bearer {_session_token}"


def _rpc(procedure: str, body: dict) -> dict:
    """Execute a ConnectRPC unary call using the Connect JSON protocol."""
    url = f"{DIAG_BASE_URL}/diag.v1.DiagramService/{procedure}"
    headers = {"Content-Type": "application/json", "Authorization": _get_auth_header()}
    clean_body = {k: v for k, v in body.items() if v is not None}
    resp = httpx.post(url, json=clean_body, headers=headers, timeout=30)
    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("message") or data.get("error") or resp.text
        raise RuntimeError(f"RPC {procedure} failed (HTTP {resp.status_code}): {error_msg}")
    return data


def _rest_get(path: str) -> dict:
    resp = httpx.get(f"{DIAG_BASE_URL}{path}", headers={"Authorization": _get_auth_header()}, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} failed ({resp.status_code}): {resp.text}")
    return resp.json()


# ── Tools ──────────────────────────────────────────────────────────────────────


@mcp.tool
def get_org() -> dict:
    """
    Retrieve the current user's organisation slug and name.

    Populates the Redis slug → UUID cache for this org so all subsequent
    tool calls can resolve it without extra network requests.

    Returns a dict with:
      - slug: the org slug (pass this as org_slug to all other tools)
      - name: the organisation's display name
    """
    org = _rest_get("/api/org")
    slug: str = org["slug"]
    uuid: str = org["id"]
    cache_org(slug, uuid)
    return {"slug": slug, "name": org.get("name")}


@mcp.tool
def create_diagram(org_slug: str, name: str, description: Optional[str] = None, level_label: Optional[str] = None, parent_diagram_slug: Optional[str] = None) -> dict:
    """
    Create a new diagram (canvas) in the given organisation.

    Args:
        org_slug: Slug of the organisation. Obtain with get_org().
        name: Human-readable name of the diagram (e.g. "Core", "Data Ingestion", ).
        description: Optional longer description of what this diagram represents.
        level_label: Optional abstraction level (e.g. "Package", "Class", "Library", "Component", "Framework").
        parent_diagram: Optional but necessary to show the abstraction level of the diagram.
                             If not provided, the diagram will be created at the root level.


    Returns:
        A dict with:
          - diagram: use this as diagram in add_node and connect_nodes
          - name, description, level_label, created_at, updated_at
    """
    org_uuid = resolve_org_slug(org_slug)
    parent_uuid = resolve_diagram_slug(org_slug, parent_diagram_slug) if parent_diagram_slug else None

    result = _rpc("CreateDiagram", {"orgId": org_uuid, "name": name, "description": description, "levelLabel": level_label, "parentDiagramId": parent_uuid})

    diagram = result.get("diagram", result)
    diagram_uuid: str = diagram["id"]

    base_slug = slugify(name)
    diagram_slug = _unique_slug(base_slug, lambda s: _diagram_slug_exists(org_slug, s))
    cache_diagram(org_slug, diagram_slug, diagram_uuid)

    return {
        "diagram": diagram_slug,
        "name": diagram.get("name"),
        "description": diagram.get("description"),
        "level_label": diagram.get("levelLabel"),
        "created_at": diagram.get("createdAt"),
        "updated_at": diagram.get("updatedAt"),
    }


@mcp.tool
def add_node(
    org_slug: str,
    diagram: str,
    name: str,
    type: str,
    description: Optional[str] = None,
    technology: Optional[str] = None,
    url: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Create a node to represent an object or an entity in the codebase and place it on a diagram.
    At high or root level these could be a  Repository, Package, Library, Framework, Microservice, Module, Application
    At medium level these could be a File, Class, Interface, Function, Method, Component, Actor
    At low level these could be a Variable, Constant, Data Structure, Queue, Field, Endpoint


    Args:
        org_slug: Slug of the organisation. Obtain with get_org().
        diagram: Target diagram. Obtain from create_diagram().
        name: Human-readable, recognizable identifier of this node. Required.
        type: Node type: "class", "function", "module", "method", "database",
              "queue", "actor", "interface", "object", "component", "system", "container".
              Required.
        description: Optional description of this node, its role or function details go here.
        technology: Optional primary technology if applicable (e.g. "Go", "PostgreSQL", "React").
        url: Optional URL or relative path within the codebase.
        tags: Optional freeform tags (e.g. ["backend", "critical"]).

    Returns:
        A dict with:
          - node: use this as source_node / target_node in connect_nodes
          - name, type, description, technology, tags
    """
    org_uuid = resolve_org_slug(org_slug)
    diagram_uuid = resolve_diagram_slug(org_slug, diagram)

    result = _rpc(
        "AddNode", {"orgId": org_uuid, "diagramId": diagram_uuid, "name": name, "type": type, "description": description, "technology": technology, "url": url, "tags": tags}
    )

    node_object = result.get("node", result)
    node_uuid: str = node_object["id"]

    base_slug = slugify(name)
    node_slug = _unique_slug(base_slug, lambda s: _node_slug_exists(org_slug, s))
    cache_node(org_slug, node_slug, node_uuid)

    return {
        "node": node_slug,
        "name": node_object.get("name"),
        "type": node_object.get("type"),
        "description": node_object.get("description"),
        "technology": node_object.get("technology"),
        "tags": node_object.get("tags"),
    }


@mcp.tool
def connect_nodes(
    org_slug: str,
    diagram: str,
    source_node: str,
    target_node: str,
    label: Optional[str] = None,
    description: Optional[str] = None,
    direction: Optional[str] = None,
    url: Optional[str] = None,
) -> dict:
    """
    Draw a directed edge between two nodes that are placed on the same diagram.

    Both nodes must already exist on the diagram (added via add_node).

    Args:
        org_slug: Slug of the organisation.
        diagram: The diagram containing both nodes.
        source_node: Source node identifier (from add_node).
        target_node: Target node identifier (from add_node).
        label: Short label shown on the edge (e.g. "fetch data", "depends_on", "write", "calls", "publish events").
        description: Longer description of the relationship.
        direction: "forward" (default, source→target), "backward", "both", or "none".
        url: Optional remote url or relative path to the related resource.

    Returns:
        A dict with edge details.
    """
    diagram_uuid = resolve_diagram_slug(org_slug, diagram)
    source_uuid = resolve_node_slug(org_slug, source_node)
    target_uuid = resolve_node_slug(org_slug, target_node)

    result = _rpc(
        "ConnectNodes",
        {"diagramId": diagram_uuid, "sourceNodeId": source_uuid, "targetNodeId": target_uuid, "label": label, "description": description, "direction": direction, "url": url},
    )

    edge = result.get("edge", result)
    return {
        "diagram": diagram,
        "source_node": source_node,
        "target_node": target_node,
        "label": edge.get("label"),
        "description": edge.get("description"),
        "direction": edge.get("direction"),
        "created_at": edge.get("createdAt"),
    }


@mcp.prompt(name="create_codebase_diagram", description="Create a diagram based on the structure of a code repository.")
def create_codebase_diagram(
    repo_path: str = Field(description="Local or remote path to the code repository."),
    username: str = Field(default="admin", description="Username for authentication (if needed)."),
    password: str = Field(default="password", description="Password for authentication (if needed)."),
) -> str:
    instruction = f"""
    # Role:
    You are a senior software architect and documentation specialist.
    Use your expertise to create clear, concise architecture diagrams that capture the core structure and key flows of our codebase.
    Then use the tool available in your current environment to create the diagrams programmatically.
    A typical step-by-step process is like this:

    1 - get org_slug from tool get_org() with username: {username} and password: {password}.
    2 - use tool create_diagram().
    3 - use tool add_node() for the crucial objects you identify in the codebase (business logic, main data flows, key decisions, public APIs, critical integrations).
    4 - use tool connect_nodes() to define the relationships between those objects.
    5 - Repeat steps 2–4 to create multiple diagrams for different views or levels of abstraction, reusing nodes where they appear in multiple contexts.

    Your mission is to explore this codebase in {repo_path} using the tools available in your current environment (file browsing, search, read-file, repo indexing).
    Prioritize reads: Start with the most critical files first (entry points, core modules, configs, database models, major features).

    Produce 5–25 topological diagrams (one per key flow or responsibility) that help any new team member understand the real core of our codebase in under 5 minutes.
    Start with the highest-level, most abstract diagrams (e.g. main user flow, core microservices, critical domain).
    Cluster the complex systems into at max 20 nodes. Then create a separate diagram with same object to drill down into details.
    Focus exclusively on crucial objects — business logic, main data flows, key decisions, public APIs, and critical integrations.
    Ignore everything else: no logging, no config files, no error handlers, no tests, no boilerplate, no generated code.

    Choose 1–10 Root-Level Diagrams. Decide on the number based on the complexity of the system and how many distinct “views” are needed to cover the essentials.

    Pick the most important “views” of the system. Good examples:
    Main User Flow (e.g. “Order Placement”)
    Core Microservice Architecture
    Authentication Flow
    Data Pipeline / Background Job
    External Integration (e.g. Payment Gateway)
    Startup / Boot Sequence


    For each diagram, build 3–20 nodes total
    At high or root level these could be a Repository, Package, Library, Framework, Microservice, Module, Application
    At medium level these could be a File, Class, Interface, Function, Method, Component, Actor
    At low level these could be a Variable, Constant, Data Structure, Queue, Field, Endpoint

    If a node is referenced multiple times, create it once and reuse it across diagrams to show how different parts of the system connect to the same core concepts.

    Connect the nodes with a clear edge label to represent the interaction (e.g. "fetch data", "depends_on", "write", "calls", "publish events")

    These could be relationships like: contains, depends_on, imports, calls, inherits_from, returns, uses, references.

    Title each diagram clearly (e.g. “Order Placement Flow”)
    Use 1-sentence caption for description of each diagram/node/edge.
    """
    return instruction


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8000"))
    host = os.getenv("MCP_HOST", "0.0.0.0")
    mcp.run(transport="http", host=host, port=port)
