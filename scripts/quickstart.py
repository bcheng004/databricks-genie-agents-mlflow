"""Provision the Genie MLflow app's UC-managed experiment and wire up config.

This is the CLI port of what used to be the app's Home/Setup page. It:

  1. Validates (or interactively creates) a Databricks CLI profile.
  2. Detects a SQL warehouse and creates the UC schema for trace storage.
  3. Creates (or reuses) a Unity Catalog–managed MLflow experiment whose trace
     location points at that catalog/schema/table-prefix.
  4. Writes the resolved config into .env, app/app.yaml (what the deployed app
     reads at runtime), and databricks.yml bundle variable defaults.

Reruns are idempotent — an experiment recorded in .env is reused as-is.

Run with no flags to be prompted for the profile, experiment path, catalog,
schema, and table prefix (each prompt defaults to the cached .env value):

    uv run quickstart

Or pass any subset of flags to skip those prompts (useful for CI):

    uv run quickstart --profile mlflow-workshop \
        --experiment-name /Workspace/Shared/genie-eval-traces \
        --catalog main --schema genie_traces --table-prefix evals \
        --warehouse-id abc123
"""

from __future__ import annotations

import argparse
import os
import sys

from ._common import (
    ensure_env_file,
    get_env_value,
    prompt_value,
    resolve_profile,
    set_app_yaml_env,
    set_bundle_variable_default,
    update_env_file,
    validate_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile to authenticate with. "
        "If omitted, you'll be prompted to choose one interactively.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment path (workspace path). Created if it does not "
        "exist. Prompted for interactively if omitted.",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Unity Catalog catalog for trace storage. Prompted if omitted.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Unity Catalog schema for trace storage. Prompted if omitted.",
    )
    parser.add_argument(
        "--table-prefix",
        default=None,
        help="Trace table prefix. Prompted if omitted.",
    )
    parser.add_argument(
        "--warehouse-id",
        default=get_env_value("MLFLOW_TRACING_SQL_WAREHOUSE_ID"),
        help="SQL warehouse ID. Auto-detected from the workspace if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the experiment even if MLFLOW_EXPERIMENT_ID is cached.",
    )
    return parser.parse_args()


def detect_warehouse_id(w, provided: str | None) -> str:
    """Return the provided warehouse, else a Serverless Starter, else the first."""
    if provided:
        return provided
    warehouses = list(w.warehouses.list())
    if not warehouses:
        sys.exit(
            "No SQL warehouse found. Create one (or pass --warehouse-id) and rerun."
        )
    wh = next(
        (
            x
            for x in warehouses
            if "Serverless Starter Warehouse" in (x.name or "")
            and x.enable_serverless_compute
        ),
        warehouses[0],
    )
    print(f"  Using SQL warehouse: {wh.name} ({wh.id})")
    return wh.id


def create_uc_schema(w, warehouse_id: str, catalog: str, schema: str) -> None:
    print(f"  Ensuring UC schema {catalog}.{schema} …")
    try:
        w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}",
        )
    except Exception as exc:  # noqa: BLE001 — surface but don't abort
        print(f"    (warning) could not create schema: {exc}")


def create_experiment(args) -> tuple[str, str]:
    """Create or reuse the UC-managed experiment; return (name, id)."""
    import mlflow
    from mlflow.entities.trace_location import UnityCatalog

    mlflow.set_tracking_uri("databricks")

    # Resolve by the REQUESTED name (not a cached ID), so typing a new
    # experiment path on a rerun actually takes effect. An experiment already
    # at that path is reused as-is; --force recreates it.
    if not args.force:
        existing = mlflow.get_experiment_by_name(args.experiment_name)
        if existing is not None:
            print(
                f"  Reusing existing experiment {existing.name} "
                f"({existing.experiment_id})."
            )
            return existing.name, existing.experiment_id

    print(f"  Configuring experiment {args.experiment_name} …")
    exp = mlflow.set_experiment(
        experiment_name=args.experiment_name,
        trace_location=UnityCatalog(
            catalog_name=args.catalog,
            schema_name=args.schema,
            table_prefix=args.table_prefix or "evals",
        ),
    )
    print(
        f"    OTel spans table: {exp.trace_location.full_otel_spans_table_name}"
    )
    return args.experiment_name, exp.experiment_id


def main() -> None:
    args = parse_args()
    ensure_env_file()

    profile = resolve_profile(args.profile)

    print(f"[1/4] Validating Databricks profile '{profile}' …")
    if not validate_profile(profile):
        sys.exit(
            f"Profile '{profile}' is not authenticated. Run:\n"
            f"  databricks auth login --profile {profile}"
        )
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    # Resolve UC trace-location inputs (flag → prompt with cached/fallback default).
    args.experiment_name = prompt_value(
        "Experiment path",
        args.experiment_name,
        "MLFLOW_EXPERIMENT_NAME",
        "/Workspace/Shared/genie-eval-traces",
    )
    args.catalog = prompt_value("Catalog", args.catalog, "UC_CATALOG", "main")
    args.schema = prompt_value("Schema", args.schema, "UC_SCHEMA", "genie_traces")
    args.table_prefix = prompt_value(
        "Table prefix", args.table_prefix, "UC_TABLE_PREFIX", "evals"
    )

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=profile)

    print("[2/4] Resolving SQL warehouse and UC schema …")
    warehouse_id = detect_warehouse_id(w, args.warehouse_id)
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
    create_uc_schema(w, warehouse_id, args.catalog, args.schema)

    print("[3/4] Creating / reusing UC-managed MLflow experiment …")
    exp_name, exp_id = create_experiment(args)

    print("[4/4] Writing config to .env, app/app.yaml, databricks.yml …")
    env_updates = {
        "DATABRICKS_CONFIG_PROFILE": profile,
        "MLFLOW_EXPERIMENT_NAME": exp_name,
        "MLFLOW_EXPERIMENT_ID": exp_id,
        "UC_CATALOG": args.catalog,
        "UC_SCHEMA": args.schema,
        "UC_TABLE_PREFIX": args.table_prefix,
        "MLFLOW_TRACING_SQL_WAREHOUSE_ID": warehouse_id,
    }
    for key, value in env_updates.items():
        update_env_file(key, value)

    set_app_yaml_env(
        {
            "MLFLOW_EXPERIMENT_NAME": exp_name,
            "MLFLOW_TRACING_SQL_WAREHOUSE_ID": warehouse_id,
        }
    )
    set_bundle_variable_default("warehouse_id", warehouse_id)

    print(
        "\nDone. Experiment ready:\n"
        f"  name: {exp_name}\n"
        f"  id:   {exp_id}\n\n"
        "Next: `uv run create-genie-agent` to attach a Genie agent, "
        "then deploy with `databricks bundle deploy`."
    )


if __name__ == "__main__":
    main()
