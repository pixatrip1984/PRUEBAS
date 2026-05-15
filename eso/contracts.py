"""Lightweight validation for ESO JSON contracts.

This is intentionally small and dependency-free. It supports the subset of
JSON Schema used by the versioned schemas in ./schemas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_BY_VERSION = {
    "eso.asset_dissect.v1": "asset_dissect.v1.json",
    "eso.asset_universe.v1": "asset_universe.v1.json",
    "eso.benchmark_result.v1": "benchmark_result.v1.json",
    "eso.benchmark_suite.v1": "benchmark_suite.v1.json",
    "eso.model_run.v1": "model_run.v1.json",
    "eso.model_handoff.v1": "model_handoff.v1.json",
    "eso.report.v1": "report.v1.json",
    "eso.universe_dissect.v1": "universe_dissect.v1.json",
}


def load_schema(schema_version: str) -> dict:
    if schema_version not in SCHEMA_BY_VERSION:
        raise ValueError(f"unknown schema version: {schema_version}")
    path = SCHEMA_DIR / SCHEMA_BY_VERSION[schema_version]
    return json.loads(path.read_text(encoding="utf-8"))


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


def _validate_node(value: Any, schema: dict, path: str, errors: list[str]) -> None:
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}, got {value!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}, got {value!r}")
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
        return

    if expected_type == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: missing required key")
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            if key in value:
                _validate_node(value[key], sub_schema, f"{path}.{key}", errors)
    elif expected_type == "array" and "items" in schema:
        item_schema = schema["items"]
        for i, item in enumerate(value):
            _validate_node(item, item_schema, f"{path}[{i}]", errors)


def validate_contract(payload: dict, schema_version: str | None = None) -> list[str]:
    """Return validation errors for a payload, or [] when valid."""
    schema_version = schema_version or payload.get("schema_version")
    schema = load_schema(str(schema_version))
    errors: list[str] = []
    _validate_node(payload, schema, "$", errors)
    return errors


def assert_valid_contract(payload: dict, schema_version: str | None = None) -> None:
    errors = validate_contract(payload, schema_version=schema_version)
    if errors:
        raise ValueError("contract validation failed: " + "; ".join(errors))
