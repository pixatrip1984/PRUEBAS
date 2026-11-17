"""SQLite-backed relational memory for ESO mathematical experiments."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"), default=str)


def _decode_json(value: str | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


DEFAULT_ARSENAL: tuple[dict[str, Any], ...] = (
    {"method_id": "circle", "name": "Circle S1", "family": "periodic_compact", "algebraic_group": "SO(2)", "category_tags": ["manifold", "compact", "periodic"], "description": "One-dimensional compact periodic manifold."},
    {"method_id": "sphere2", "name": "Sphere S2", "family": "curved_spherical", "algebraic_group": "SO(3)", "category_tags": ["manifold", "compact", "positive_curvature"], "description": "Two-dimensional positively curved compact manifold."},
    {"method_id": "sphere3", "name": "Sphere S3", "family": "curved_spherical", "algebraic_group": "SO(4)", "category_tags": ["manifold", "compact", "positive_curvature"], "description": "Three-dimensional positively curved compact manifold."},
    {"method_id": "sphere4", "name": "Sphere S4", "family": "curved_spherical", "algebraic_group": "SO(5)", "category_tags": ["manifold", "compact", "positive_curvature"], "description": "Four-dimensional positively curved compact manifold."},
    {"method_id": "torus2", "name": "Torus T2", "family": "periodic_compact", "algebraic_group": "U(1)^2", "category_tags": ["manifold", "compact", "periodic"], "description": "Two independent compact periodic coordinates."},
    {"method_id": "torus3", "name": "Torus T3", "family": "periodic_compact", "algebraic_group": "U(1)^3", "category_tags": ["manifold", "compact", "periodic"], "description": "Three independent compact periodic coordinates."},
    {"method_id": "cylinder", "name": "Cylinder S1 x R", "family": "non_compact_product", "algebraic_group": "SO(2)xR", "category_tags": ["manifold", "periodic", "non_compact", "product"], "description": "Periodic coordinate with an open drift axis."},
    {"method_id": "cone", "name": "Cone", "family": "singular_transition", "algebraic_group": "R+xSO(2)", "category_tags": ["manifold", "non_compact", "singular"], "description": "Degenerate geometry with an apex transition."},
    {"method_id": "klein_bottle", "name": "Klein bottle", "family": "topological_invariant", "algebraic_group": "non_orientable_compact", "category_tags": ["manifold", "compact", "non_orientable", "topological"], "description": "Compact non-orientable surface."},
    {"method_id": "mobius", "name": "Mobius band", "family": "topological_invariant", "algebraic_group": "non_orientable_boundary", "category_tags": ["manifold", "non_compact", "non_orientable", "topological"], "description": "Non-orientable surface with boundary."},
    {"method_id": "hyperbolic", "name": "Hyperbolic disk H2", "family": "curved_hyperbolic", "algebraic_group": "SO(2,1)", "category_tags": ["manifold", "non_compact", "negative_curvature"], "description": "Negatively curved Poincare disk geometry."},
    {"method_id": "h2_r", "name": "H2 x R", "family": "non_compact_product", "algebraic_group": "SO(2,1)xR", "category_tags": ["manifold", "non_compact", "negative_curvature", "product"], "description": "Hyperbolic state geometry with an open drift coordinate."},
    {"method_id": "plane2d", "name": "Euclidean plane R2", "family": "flat_euclidean", "algebraic_group": "ISO(2)", "category_tags": ["manifold", "flat", "non_compact"], "description": "Two-dimensional Euclidean baseline."},
    {"method_id": "plane3d", "name": "Euclidean space R3", "family": "flat_euclidean", "algebraic_group": "ISO(3)", "category_tags": ["manifold", "flat", "non_compact"], "description": "Three-dimensional Euclidean baseline."},
    {"method_id": "plane4d", "name": "Euclidean space R4", "family": "flat_euclidean", "algebraic_group": "ISO(4)", "category_tags": ["manifold", "flat", "non_compact"], "description": "Four-dimensional Euclidean baseline."},
    {"method_id": "s1_r2", "name": "S1 x R2", "family": "non_compact_product", "algebraic_group": "SO(2)xR^2", "category_tags": ["manifold", "periodic", "non_compact", "product"], "description": "One periodic mode with a two-dimensional drift plane."},
    {"method_id": "s2_r", "name": "S2 x R", "family": "non_compact_product", "algebraic_group": "SO(3)xR", "category_tags": ["manifold", "positive_curvature", "non_compact", "product"], "description": "Spherical state shell with linear drift."},
    {"method_id": "t2_r", "name": "T2 x R", "family": "non_compact_product", "algebraic_group": "U(1)^2xR", "category_tags": ["manifold", "periodic", "non_compact", "product"], "description": "Two periodic modes with a linear drift coordinate."},
    {"method_id": "s2_s1", "name": "S2 x S1", "family": "periodic_compact", "algebraic_group": "SO(3)xSO(2)", "category_tags": ["manifold", "compact", "periodic", "product"], "description": "Compact spherical state shell with periodic modulation."},
    {"method_id": "spectral_causal", "name": "Spectral causal operator", "family": "spectral_causal", "algebraic_group": "FourierxDAG", "category_tags": ["operator", "spectral", "causal"], "description": "Frequency-domain structure with directed causal ordering."},
    {"method_id": "persistent_homology", "name": "Persistent homology", "family": "topological_invariant", "algebraic_group": "Homology", "category_tags": ["operator", "topological", "invariant"], "description": "Multi-scale homological persistence operator."},
    {"method_id": "microstructure_flow", "name": "Microstructure flow", "family": "microstructure_flow", "algebraic_group": "OrderFlow", "category_tags": ["operator", "microstructure", "flow"], "description": "Order-flow and liquidity-structure operator."},
    {"method_id": "cascade_branching", "name": "Cascade branching", "family": "cascade_branching", "algebraic_group": "Hawkes", "category_tags": ["operator", "cascade", "branching"], "description": "Self-exciting branching and avalanche operator."},
)


class RelationalLabStore:
    """Normalized local SQLite store for ESO datasets, methods, and experiments."""

    def __init__(self, path: str | Path = "experiments/eso_lab.sqlite3") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets_configs (
                config_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                scale_s TEXT NOT NULL,
                space_dim_j INTEGER NOT NULL,
                transform_config_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS math_arsenal (
                method_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                family TEXT NOT NULL,
                algebraic_group TEXT NOT NULL,
                category_tags_json TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                config_id TEXT REFERENCES datasets_configs(config_id),
                method_id TEXT NOT NULL REFERENCES math_arsenal(method_id),
                signature_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                relative_to_flat REAL,
                oos_stability REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS method_crossovers (
                crossover_id TEXT PRIMARY KEY,
                parent_method_a TEXT NOT NULL REFERENCES math_arsenal(method_id),
                parent_method_b TEXT NOT NULL REFERENCES math_arsenal(method_id),
                emergent_family TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK (parent_method_a <> parent_method_b)
            );

            CREATE TABLE IF NOT EXISTS structural_radiographies (
                radiography_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                config_id TEXT REFERENCES datasets_configs(config_id),
                summary_statement TEXT NOT NULL,
                radiography_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_experiments_method ON experiments(method_id);
            CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id);
            CREATE INDEX IF NOT EXISTS idx_arsenal_family ON math_arsenal(family);
            CREATE INDEX IF NOT EXISTS idx_crossovers_status ON method_crossovers(status);
            CREATE INDEX IF NOT EXISTS idx_radiographies_dataset ON structural_radiographies(dataset_id);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "RelationalLabStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and self._conn is not None:
            self._conn.commit()
        elif self._conn is not None:
            self._conn.rollback()
        self.close()

    def register_dataset_config(
        self,
        dataset_id: str,
        scale_s: str,
        space_dim_j: int,
        transform_config: Mapping[str, Any] | None = None,
        properties: Mapping[str, Any] | None = None,
        *,
        config_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        config_id = config_id or f"cfg-{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO datasets_configs
              (config_id, dataset_id, scale_s, space_dim_j, transform_config_json, properties_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(config_id) DO UPDATE SET
              dataset_id=excluded.dataset_id,
              scale_s=excluded.scale_s,
              space_dim_j=excluded.space_dim_j,
              transform_config_json=excluded.transform_config_json,
              properties_json=excluded.properties_json
            """,
            (
                config_id,
                dataset_id,
                scale_s,
                int(space_dim_j),
                _json(transform_config),
                _json(properties),
                created_at or _utc_now(),
            ),
        )
        self._conn.commit()
        return config_id

    def upsert_method(
        self,
        method_id: str,
        name: str,
        family: str,
        algebraic_group: str,
        category_tags: Iterable[str] | None = None,
        description: str = "",
        *,
        created_at: str | None = None,
    ) -> str:
        tags = list(category_tags or [])
        self._conn.execute(
            """
            INSERT INTO math_arsenal
              (method_id, name, family, algebraic_group, category_tags_json, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(method_id) DO UPDATE SET
              name=excluded.name,
              family=excluded.family,
              algebraic_group=excluded.algebraic_group,
              category_tags_json=excluded.category_tags_json,
              description=excluded.description
            """,
            (method_id, name, family, algebraic_group, _json(tags), description, created_at or _utc_now()),
        )
        self._conn.commit()
        return method_id

    def seed_default_arsenal(self) -> int:
        for spec in DEFAULT_ARSENAL:
            self.upsert_method(**spec)
        return len(DEFAULT_ARSENAL)

    def list_methods(self, family: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM math_arsenal"
        params: tuple[Any, ...] = ()
        if family is not None:
            sql += " WHERE family = ?"
            params = (family,)
        sql += " ORDER BY method_id"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._method_row(row) for row in rows]

    def get_method(self, method_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM math_arsenal WHERE method_id = ?", (method_id,)).fetchone()
        return self._method_row(row) if row else None

    def record_experiment(
        self,
        dataset_id: str,
        method_id: str,
        signature: Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        config_id: str | None = None,
        run_id: str | None = None,
        relative_to_flat: float | None = None,
        oos_stability: float | None = None,
        created_at: str | None = None,
    ) -> str:
        run_id = run_id or f"run-{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO experiments
              (run_id, dataset_id, config_id, method_id, signature_json, metrics_json,
               relative_to_flat, oos_stability, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                dataset_id,
                config_id,
                method_id,
                _json(signature),
                _json(metrics),
                relative_to_flat,
                oos_stability,
                created_at or _utc_now(),
            ),
        )
        self._conn.commit()
        return run_id

    def query_experiments(
        self,
        *,
        dataset_id: str | None = None,
        method_id: str | None = None,
        family: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_id is not None:
            clauses.append("e.dataset_id = ?")
            params.append(dataset_id)
        if method_id is not None:
            clauses.append("e.method_id = ?")
            params.append(method_id)
        if family is not None:
            clauses.append("m.family = ?")
            params.append(family)
        sql = """
            SELECT e.*, m.family, m.algebraic_group, m.name AS method_name
            FROM experiments e JOIN math_arsenal m ON m.method_id = e.method_id
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY e.created_at, e.run_id"
        return [self._experiment_row(row) for row in self._conn.execute(sql, params).fetchall()]

    def query_family_performance(self, family: str | None = None) -> list[dict[str, Any]]:
        clauses = ""
        params: tuple[Any, ...] = ()
        if family is not None:
            clauses = " WHERE m.family = ?"
            params = (family,)
        rows = self._conn.execute(
            """
            SELECT m.family, e.relative_to_flat, e.oos_stability, e.metrics_json
            FROM experiments e
            JOIN math_arsenal m ON m.method_id = e.method_id
            """ + clauses + " ORDER BY m.family, e.created_at, e.run_id",
            params,
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(str(row["family"]), []).append(row)

        output: list[dict[str, Any]] = []
        for family_name in sorted(grouped):
            family_rows = grouped[family_name]
            relatives = [float(r["relative_to_flat"]) for r in family_rows if r["relative_to_flat"] is not None]
            stabilities = [float(r["oos_stability"]) for r in family_rows if r["oos_stability"] is not None]
            metric_scores: list[float] = []
            for row in family_rows:
                metrics = _decode_json(row["metrics_json"], {})
                for key in ("score", "quality", "objective", "relative_to_flat"):
                    value = metrics.get(key) if isinstance(metrics, dict) else None
                    if isinstance(value, (int, float)):
                        metric_scores.append(float(value))
                        break
            output.append(
                {
                    "family": family_name,
                    "experiment_count": len(family_rows),
                    "mean_relative_to_flat": sum(relatives) / len(relatives) if relatives else None,
                    "mean_oos_stability": sum(stabilities) / len(stabilities) if stabilities else None,
                    "mean_metric_score": sum(metric_scores) / len(metric_scores) if metric_scores else None,
                }
            )
        return output

    def record_crossover(
        self,
        parent_method_a: str,
        parent_method_b: str,
        emergent_family: str,
        hypothesis: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        status: str = "proposed",
        crossover_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        if parent_method_a == parent_method_b:
            raise ValueError("crossover parents must be distinct")
        crossover_id = crossover_id or f"xover-{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO method_crossovers
              (crossover_id, parent_method_a, parent_method_b, emergent_family,
               hypothesis, metrics_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(crossover_id) DO UPDATE SET
              emergent_family=excluded.emergent_family,
              hypothesis=excluded.hypothesis,
              metrics_json=excluded.metrics_json,
              status=excluded.status
            """,
            (
                crossover_id,
                parent_method_a,
                parent_method_b,
                emergent_family,
                hypothesis,
                _json(metrics),
                status,
                created_at or _utc_now(),
            ),
        )
        self._conn.commit()
        return crossover_id

    def query_crossovers(
        self,
        *,
        status: str | None = None,
        emergent_family: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if emergent_family is not None:
            clauses.append("emergent_family = ?")
            params.append(emergent_family)
        sql = "SELECT * FROM method_crossovers"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, crossover_id"
        output = []
        for row in self._conn.execute(sql, params).fetchall():
            item = dict(row)
            item["metrics"] = _decode_json(item.pop("metrics_json"), {})
            output.append(item)
        return output

    def record_radiography(
        self,
        dataset_id: str,
        summary_statement: str,
        radiography: Mapping[str, Any],
        *,
        config_id: str | None = None,
        radiography_id: str | None = None,
        created_at: str | None = None,
    ) -> str:
        radiography_id = radiography_id or f"rad-{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO structural_radiographies
              (radiography_id, dataset_id, config_id, summary_statement, radiography_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                radiography_id,
                dataset_id,
                config_id,
                summary_statement,
                _json(radiography),
                created_at or _utc_now(),
            ),
        )
        self._conn.commit()
        return radiography_id

    def query_radiographies(
        self,
        *,
        dataset_id: str | None = None,
        config_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_id is not None:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if config_id is not None:
            clauses.append("config_id = ?")
            params.append(config_id)
        sql = "SELECT * FROM structural_radiographies"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at, radiography_id"
        output = []
        for row in self._conn.execute(sql, params).fetchall():
            item = dict(row)
            item["radiography"] = _decode_json(item.pop("radiography_json"), {})
            output.append(item)
        return output

    @staticmethod
    def _method_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["category_tags"] = _decode_json(item.pop("category_tags_json"), [])
        return item

    @staticmethod
    def _experiment_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["signature"] = _decode_json(item.pop("signature_json"), {})
        item["metrics"] = _decode_json(item.pop("metrics_json"), {})
        return item
