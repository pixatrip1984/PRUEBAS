import sqlite3

import pytest

from eso.db import RelationalLabStore
from eso.meta.soft_selector import FamilyPrioritizedSelector, MethodCrossoverGenerator


def _diagnosis(*, periodic=False, chaotic=False, dimension=2.0):
    return {
        "dimension": {"consensus_dimension": dimension},
        "stationarity": {"label": "stationary"},
        "lyapunov": {"value": 0.2 if chaotic else -0.1, "chaotic_hint": chaotic},
        "symmetries": {"periodic_hint": periodic, "dominant_frequency": {"frequency": 0.125}},
    }


def test_schema_creation_and_default_arsenal(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "datasets_configs",
            "math_arsenal",
            "experiments",
            "method_crossovers",
            "structural_radiographies",
        } <= tables
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.seed_default_arsenal() >= 23
        method_ids = {row["method_id"] for row in store.list_methods()}
        assert {"circle", "hyperbolic", "torus3", "spectral_causal", "cascade_branching"} <= method_ids


def test_experiment_persistence_and_relational_family_query(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        store.seed_default_arsenal()
        config_id = store.register_dataset_config(
            "btc-1h",
            "1h",
            3,
            {"returns": True},
            {"stationarity": "mixed"},
            config_id="cfg-btc",
        )
        run_id = store.record_experiment(
            "btc-1h",
            "hyperbolic",
            {"dim_consensus": 2.2},
            {"score": 0.8},
            config_id=config_id,
            run_id="run-1",
            relative_to_flat=0.35,
            oos_stability=0.9,
        )
        assert run_id == "run-1"
        rows = store.query_experiments(dataset_id="btc-1h", family="curved_hyperbolic")
        assert len(rows) == 1
        assert rows[0]["signature"]["dim_consensus"] == pytest.approx(2.2)
        assert rows[0]["metrics"]["score"] == pytest.approx(0.8)
        perf = store.query_family_performance("curved_hyperbolic")
        assert perf[0]["mean_relative_to_flat"] == pytest.approx(0.35)
        assert perf[0]["mean_oos_stability"] == pytest.approx(0.9)

        with pytest.raises(sqlite3.IntegrityError):
            store.record_experiment("btc-1h", "missing-method", {}, {}, config_id=config_id)


def test_non_discarding_family_prioritization_has_epsilon_floor(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        store.seed_default_arsenal()
        selector = FamilyPrioritizedSelector(store, epsilon=0.02)
        candidates = ["circle", "hyperbolic", "plane2d", "spectral_causal", "cascade_branching"]
        ranked = selector.prioritize(_diagnosis(periodic=True, dimension=1.1), candidates)
        weights = [row["probability"] for row in ranked]
        assert ranked[0]["family"] == "periodic_compact"
        assert len(ranked) == len(candidates)
        assert min(weights) >= 0.02
        assert sum(weights) == pytest.approx(1.0)
        assert set(selector.selection_weights(_diagnosis(periodic=True), candidates)) == set(candidates)


def test_historical_performance_reinforces_family_without_discarding(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        store.seed_default_arsenal()
        selector = FamilyPrioritizedSelector(store, epsilon=0.01)
        candidates = ["sphere2", "hyperbolic"]
        before = {row["method_id"]: row for row in selector.prioritize(_diagnosis(), candidates)}

        for index in range(4):
            store.record_experiment(
                "synthetic",
                "hyperbolic",
                {"dim_consensus": 2.0},
                {"score": 1.0},
                run_id=f"hist-{index}",
                relative_to_flat=1.0,
                oos_stability=1.0,
            )
        after = {row["method_id"]: row for row in selector.prioritize(_diagnosis(), candidates)}
        assert after["hyperbolic"]["historical_score"] > before["hyperbolic"]["historical_score"]
        assert after["hyperbolic"]["probability"] > before["hyperbolic"]["probability"]
        assert after["sphere2"]["probability"] >= selector.epsilon


def test_cross_family_crossover_generation_is_deterministic_and_persistent(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        store.seed_default_arsenal()
        generator = MethodCrossoverGenerator(store)
        candidates = ["circle", "hyperbolic", "spectral_causal", "cascade_branching"]
        first = generator.propose_crossovers(candidates, limit=3)
        second = generator.propose_crossovers(candidates, limit=3)
        assert first == second
        assert first
        assert all(item["parent_method_a"] != item["parent_method_b"] for item in first)
        families = {method["method_id"]: method["family"] for method in store.list_methods()}
        assert all(
            families[item["parent_method_a"]] != families[item["parent_method_b"]]
            for item in first
        )

        persisted = generator.propose_crossovers(candidates, limit=2, persist=True)
        queried = generator.query_crossovers(status="proposed")
        assert {item["crossover_id"] for item in persisted} <= {item["crossover_id"] for item in queried}
        assert all("relative_to_flat" in item["hypothesis"] for item in queried)


def test_radiography_round_trip(tmp_path):
    with RelationalLabStore(tmp_path / "lab.sqlite3") as store:
        config_id = store.register_dataset_config(
            "synthetic", "window-64", 2, {}, {}, config_id="cfg-rad"
        )
        radiography_id = store.record_radiography(
            "synthetic",
            "Scale window-64 in J=2 exposes a periodic compact geometry.",
            {"geometry": "periodic_compact", "properties": ["R", "P", "K"]},
            config_id=config_id,
            radiography_id="rad-1",
        )
        assert radiography_id == "rad-1"
        rows = store.query_radiographies(dataset_id="synthetic")
        assert rows[0]["radiography"]["geometry"] == "periodic_compact"
