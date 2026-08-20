"""Provision the Genie MLflow app's experiment and wire up config.

This is the CLI port of what used to be the app's Home/Setup page. It:

  1. Validates (or interactively creates) a Databricks CLI profile.
  2. For UC-backed traces, resolves a serverless SQL warehouse (prompted,
     defaulting to the configured or a detected serverless warehouse) and
     creates the UC schema for trace storage. Skipped for workspace-backed
     traces, which need no warehouse or UC tables.
  3. Creates (or reuses) the MLflow experiment — Unity Catalog–managed (trace
     location at the catalog/schema/table-prefix) when UC-backed, otherwise a
     plain workspace-backed experiment whose traces live in the MLflow backend.
  4. Writes the resolved config into .env, app/app.yaml (what the deployed app
     reads at runtime), and databricks.yml bundle variable defaults and target
     workspace hosts (resolved from the selected profile).

Reruns are idempotent — an experiment recorded in .env is reused as-is.

Run with no flags to be prompted for the profile, experiment path, whether
traces are UC-backed, and (for UC-backed) the catalog, schema, table prefix,
and serverless SQL warehouse ID (each prompt defaults to the cached .env value,
and the warehouse falls back to a detected serverless warehouse):

    uv run quickstart

Or pass any subset of flags to skip those prompts (useful for CI):

    uv run quickstart --profile mlflow-workshop \
        --experiment-name /Workspace/Shared/genie-eval-traces \
        --catalog main --schema genie_traces --table-prefix evals \
        --warehouse-id abc123

    # Workspace-backed traces (no UC tables or warehouse):
    uv run quickstart --profile mlflow-workshop --no-uc-backed \
        --experiment-name /Workspace/Shared/genie-eval-traces
"""

from __future__ import annotations

import argparse
import os
import sys

from ._common import (
    ensure_env_file,
    get_env_value,
    prompt_bool,
    prompt_value,
    resolve_profile,
    set_app_resource_enabled,
    set_app_yaml_env,
    set_app_yaml_env_enabled,
    set_bundle_target_hosts,
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
        "--uc-backed",
        dest="uc_backed",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Store traces in a Unity Catalog–managed experiment (catalog / "
        "schema / warehouse). Use --no-uc-backed for a plain workspace-backed "
        "experiment (no UC tables or warehouse). Prompted if omitted (default: "
        "UC-backed).",
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
    """Create or reuse the MLflow experiment; return (name, id).

    UC-backed (``args.uc_backed``): traces are stored in a Unity Catalog–managed
    experiment (catalog / schema / table-prefix), queried at runtime through a
    SQL warehouse. Otherwise a plain workspace-backed experiment is created —
    traces live in the MLflow backend and need no UC tables or warehouse.
    """
    import mlflow

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
    if args.uc_backed:
        from mlflow.entities.trace_location import UnityCatalog

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
    else:
        exp = mlflow.set_experiment(experiment_name=args.experiment_name)
        print("    Workspace-backed experiment (traces stored in MLflow).")
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

    args.experiment_name = prompt_value(
        "Experiment path",
        args.experiment_name,
        "MLFLOW_EXPERIMENT_NAME",
        "/Workspace/Shared/genie-eval-traces",
    )

    # UC-backed traces (catalog/schema/warehouse) vs a plain workspace-backed
    # experiment (traces in the MLflow backend, no UC tables or warehouse).
    args.uc_backed = prompt_bool(
        "UC-backed MLflow traces?", args.uc_backed, default=True
    )

    # UC trace-location inputs are only needed for the UC-backed path.
    if args.uc_backed:
        args.catalog = prompt_value("Catalog", args.catalog, "UC_CATALOG", "main")
        args.schema = prompt_value("Schema", args.schema, "UC_SCHEMA", "genie_traces")
        args.table_prefix = prompt_value(
            "Table prefix", args.table_prefix, "UC_TABLE_PREFIX", "evals"
        )

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=profile)

    warehouse_id = None
    if args.uc_backed:
        print("[2/4] Resolving serverless SQL warehouse and UC schema …")
        warehouse_id = resolve_warehouse_id(w, args.warehouse_id)
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
        create_uc_schema(w, warehouse_id, args.catalog, args.schema)
    else:
        print("[2/4] Workspace-backed traces — skipping warehouse and UC schema.")

    workspace_url, workspace_id = resolve_embed_config(w)
    print(f"  Workspace URL: {workspace_url}")
    if workspace_id:
        print(f"  Workspace ID:  {workspace_id}")

    kind = "UC-managed" if args.uc_backed else "workspace-backed"
    print(f"[3/4] Creating / reusing {kind} MLflow experiment …")
    exp_name, exp_id = create_experiment(args)

    print("[4/4] Writing config to .env, app/app.yaml, databricks.yml …")
    env_updates = {
        "DATABRICKS_CONFIG_PROFILE": profile,
        "MLFLOW_EXPERIMENT_NAME": exp_name,
        "MLFLOW_EXPERIMENT_ID": exp_id,
        "DATABRICKS_WORKSPACE_URL": workspace_url,
    }
    # UC trace-location keys / warehouse only apply to the UC-backed path.
    if args.uc_backed:
        env_updates.update(
            {
                "UC_CATALOG": args.catalog,
                "UC_SCHEMA": args.schema,
                "UC_TABLE_PREFIX": args.table_prefix,
                "MLFLOW_TRACING_SQL_WAREHOUSE_ID": warehouse_id,
            }
        )
    if workspace_id:
        env_updates["DATABRICKS_WORKSPACE_ID"] = workspace_id
    for key, value in env_updates.items():
        update_env_file(key, value)

    app_yaml_env = {
        "MLFLOW_EXPERIMENT_NAME": exp_name,
        "DATABRICKS_WORKSPACE_URL": workspace_url,
    }
    if workspace_id:
        app_yaml_env["DATABRICKS_WORKSPACE_ID"] = workspace_id
    # For UC-backed traces the app queries trace tables through a SQL warehouse,
    # so MLFLOW_TRACING_SQL_WAREHOUSE_ID must be an active env entry. Uncomment it
    # first (a prior --no-uc-backed run may have commented it) so the upsert below
    # updates the existing line instead of appending a duplicate.
    if args.uc_backed:
        set_app_yaml_env_enabled("MLFLOW_TRACING_SQL_WAREHOUSE_ID", True)
        app_yaml_env["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id
    set_app_yaml_env(app_yaml_env)
    if warehouse_id:
        set_bundle_variable_default("warehouse_id", warehouse_id)
    if workspace_url:
        set_bundle_target_hosts(workspace_url)
    # Workspace-backed traces need no warehouse. Comment out both the app's
    # sql-warehouse resource and the MLFLOW_TRACING_SQL_WAREHOUSE_ID env entry
    # (restored when UC-backed) so neither deploy binds nor runtime queries a
    # warehouse the app has no grant on.
    set_app_resource_enabled("sql-warehouse", args.uc_backed)
    if not args.uc_backed:
        set_app_yaml_env_enabled("MLFLOW_TRACING_SQL_WAREHOUSE_ID", False)

    print(
        "\nDone. Experiment ready:\n"
        f"  name: {exp_name}\n"
        f"  id:   {exp_id}\n\n"
        "Next: `uv run add-genie-agent` to attach a Genie agent, "
        "then deploy with `databricks bundle deploy`."
    )


if __name__ == "__main__":
    main()
