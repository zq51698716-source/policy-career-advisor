"""
Shared tool library — tool schemas, execution functions, and registry.

Agents declare which tools they use by name; the registry maps names to
(JSON_schema, execute_fn) pairs.  Adding a new tool means:
  1. Define schema + execute_fn in a *_tools module
  2. Register it in TOOL_REGISTRY below
  3. Add the tool name to the agent's tool list
"""

from backend.tools.search_tools import TOOL_SCHEMAS as SEARCH_SCHEMAS, EXECUTORS as SEARCH_EXECUTORS
from backend.tools.rag_tools import TOOL_SCHEMAS as RAG_SCHEMAS, EXECUTORS as RAG_EXECUTORS
from backend.tools.utility_tools import TOOL_SCHEMAS as UTIL_SCHEMAS, EXECUTORS as UTIL_EXECUTORS

# ============================================================
# Unified tool registry: name → (schema, execute_fn)
# ============================================================
TOOL_REGISTRY: dict[str, tuple[dict, callable]] = {}

def _register_module(schemas: dict, executors: dict):
    """Register a module's tools, checking for duplicates."""
    for name, schema in schemas.items():
        if name in TOOL_REGISTRY:
            raise ValueError(f"Duplicate tool name '{name}' in registry")
        executor = executors.get(name)
        if executor is None:
            raise ValueError(f"Tool '{name}' has schema but no executor")
        TOOL_REGISTRY[name] = (schema, executor)

_register_module(SEARCH_SCHEMAS, SEARCH_EXECUTORS)
_register_module(RAG_SCHEMAS, RAG_EXECUTORS)
_register_module(UTIL_SCHEMAS, UTIL_EXECUTORS)


# Convenience: get schemas for a list of tool names
def get_tool_schemas(tool_names: list[str]) -> list[dict]:
    """Return the JSON schemas for the named tools."""
    schemas = []
    for name in tool_names:
        entry = TOOL_REGISTRY.get(name)
        if entry is None:
            raise KeyError(f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}")
        schemas.append(entry[0])
    return schemas


# Convenience: execute a tool by name
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a registered tool by name, returning a JSON result string."""
    entry = TOOL_REGISTRY.get(tool_name)
    if entry is None:
        import json
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)
    return entry[1](**tool_input)


__all__ = [
    "TOOL_REGISTRY",
    "get_tool_schemas",
    "execute_tool",
]
