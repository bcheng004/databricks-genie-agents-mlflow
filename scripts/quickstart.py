"""Provision the Genie MLflow app's UC-managed experiment and wire up config.

This is the CLI port of what used to be the app's Home/Setup page. It:

  1. Validates (or interactively creates) a Databricks CLI profile.
  2. Resolves a serverless SQL warehouse (prompted, defaulting to the one
     already configured or a detected serverless warehouse) and creates the UC
     schema for trace storage.
  3. Creates (or reuses) a Unity Catalog–managed MLflow experiment whose trace
     location points at that catalog/schema/table-prefix.
  4. Writes the resolved config into .env, app/app.yaml (what the deployed app
     reads at runtime), and databricks.yml bundle variable defaults.

Reruns are idempotent — an experiment recorded in .env is reused as-is.

Run with no flags to be prompted for the profile, experiment path, catalog,
schema, table prefix, and serverless SQL warehouse ID (each prompt defaults to
the cached .env value, and the warehouse falls back to a detected serverless
warehouse):

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
        default=None,
        help="Serverless SQL warehouse ID. Prompted for interactively if "
        "omitted (defaulting to the cached value or a detected serverless "
        "warehouse).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the experiment even if MLFLOW_EXPERIMENT_ID is cached.",
    )
    return parser.parse_args()


def detect_serverless_warehouse_id(w) -> str | None:
    """Return a serverless SQL warehouse ID to offer as the prompt default.

    Prefers a "Serverless Starter Warehouse", then any serverless-enabled
    warehouse. Returns None (best effort) if none is found — the prompt then
    simply has no default and requires an explicit ID.
    """
    try:
        warehouses = list(w.warehouses.list())
    except Exception as exc:  # noqa: BLE001 — best effort; prompt still works
        print(f"    (warning) could not list SQL warehouses: {exc}")
        return None
    serverless = [x for x in warehouses if x.enable_serverless_compute]
    wh = next(
        (x for x in serverless if "Serverless Starter Warehouse" in (x.name or "")),
        serverless[0] if serverless else None,
    )
    if wh is None:
        return None
    print(f"  Detected serverless SQL warehouse: {wh.name} ({wh.id})")
    return wh.id


def resolve_warehouse_id(w, provided: str | None) -> str:
    """Return the serverless SQL warehouse ID: flag → prompt (cached/detected default)."""
    default = get_env_value("MLFLOW_TRACING_SQL_WAREHOUSE_ID")
    if provided is None and not default:
        default = detect_serverless_warehouse_id(w)
    warehouse_id = prompt_value(
        "Serverless SQL warehouse ID",
        provided,
        None,
        default or "",
    )
    if not warehouse_id:
        sys.exit(
            "A serverless SQL warehouse ID is required. Create one (or pass "
            "--warehouse-id) and rerun."
        )
    return warehouse_id


def resolve_embed_config(w) -> tuple[str, str | None]:
    """Return (workspace_url, workspace_id) used to build the Genie embed URL.

    The workspace URL comes from the client config; the workspace (org) ID is
    the `o=` query param in the embed URL. A missing ID is non-fatal — the page
    can still resolve it at runtime.
    """
    workspace_url = (w.config.host or "").rstrip("/")
    try:
        workspace_id = str(w.get_workspace_id())
    except Exception as exc:  # noqa: BLE001 — surface but don't abort
        print(f"    (warning) could not resolve workspace ID: {exc}")
        workspace_id = None
    return workspace_url, workspace_id


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

    print("[2/4] Resolving serverless SQL warehouse and UC schema …")
    warehouse_id = resolve_warehouse_id(w, args.warehouse_id)
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
    create_uc_schema(w, warehouse_id, args.catalog, args.schema)

    workspace_url, workspace_id = resolve_embed_config(w)
    print(f"  Workspace URL: {workspace_url}")
    if workspace_id:
        print(f"  Workspace ID:  {workspace_id}")

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
        "DATABRICKS_WORKSPACE_URL": workspace_url,
    }
    if workspace_id:
        env_updates["DATABRICKS_WORKSPACE_ID"] = workspace_id
    for key, value in env_updates.items():
        update_env_file(key, value)

    app_yaml_env = {
        "MLFLOW_EXPERIMENT_NAME": exp_name,
        "MLFLOW_TRACING_SQL_WAREHOUSE_ID": warehouse_id,
        "DATABRICKS_WORKSPACE_URL": workspace_url,
    }
    if workspace_id:
        app_yaml_env["DATABRICKS_WORKSPACE_ID"] = workspace_id
    set_app_yaml_env(app_yaml_env)
    set_bundle_variable_default("warehouse_id", warehouse_id)

    print(
        "\nDone. Experiment ready:\n"
        f"  name: {exp_name}\n"
        f"  id:   {exp_id}\n\n"
        "Next: `uv run add-genie-agent` to attach a Genie agent, "
        "then deploy with `databricks bundle deploy`."
    )


if __name__ == "__main__":
    main()
