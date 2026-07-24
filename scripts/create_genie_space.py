"""Attach a Genie space to the app (reuse an existing one or create a new one).

Writes the resolved GENIE_SPACE_ID into .env, app/app.yaml (read by the running
app), and the databricks.yml `genie_space_id` bundle variable default.

Run with no flags to be prompted for the profile and the space (attach an
existing one by ID, or reuse/create one by title):

    uv run create-genie-space

Or pass flags to skip the prompts (useful for CI):

    # Wire up a space you already have:
    uv run create-genie-space --space-id 01f0123456789abc

    # Reuse by title, or create if not found:
    uv run create-genie-space --title "Sales Genie" --warehouse-id abc123 \
        --table main.sales.orders --table main.sales.customers
"""

from __future__ import annotations

import argparse
import json
import sys

from ._common import (
    ensure_env_file,
    prompt_value,
    resolve_profile,
    set_app_yaml_env,
    set_bundle_variable_default,
    update_env_file,
    validate_profile,
)

_DEFAULT_DESCRIPTION = "Genie space traced and evaluated by the Genie MLflow app."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=None,
        help="Databricks CLI profile. Prompted for interactively if omitted.",
    )
    parser.add_argument(
        "--space-id",
        help="Existing Genie space ID to attach (skips lookup/creation).",
    )
    parser.add_argument("--title", help="Space title to reuse-by-name or create.")
    parser.add_argument(
        "--warehouse-id",
        default=None,
        help="SQL warehouse for a newly created space. Prompted if needed.",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        dest="tables",
        help="Fully-qualified table (catalog.schema.table) for a new space; repeatable.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Description for a newly created space.",
    )
    return parser.parse_args()


def find_space_by_title(w, title: str):
    """Return the first space whose title matches, paging through all spaces."""
    page_token = None
    while True:
        resp = w.genie.list_spaces(page_token=page_token)
        for space in resp.spaces or []:
            if (space.title or "") == title:
                return space
        page_token = resp.next_page_token
        if not page_token:
            return None


def prompt_tables(provided: list[str] | None) -> list[str]:
    """Return the table list, prompting one-per-line when not supplied via --table."""
    if provided:
        return provided
    print("Enter fully-qualified tables (catalog.schema.table), one per line.")
    print("Press Enter on a blank line when done.")
    tables: list[str] = []
    while True:
        line = input(f"  table {len(tables) + 1} (blank to finish): ").strip()
        if not line:
            if tables:
                return tables
            print("  At least one table is required.")
            continue
        tables.append(line)


def create_space(w, args) -> str:
    """Create a new Genie space, prompting for any missing inputs."""
    title = prompt_value("Space title", args.title)
    warehouse_id = prompt_value(
        "Warehouse ID", args.warehouse_id, "MLFLOW_TRACING_SQL_WAREHOUSE_ID"
    )
    tables = prompt_tables(args.tables)
    description = prompt_value(
        "Description", args.description, fallback=_DEFAULT_DESCRIPTION
    )

    serialized_space = {
        "data_sources": {"tables": [{"identifier": t} for t in tables]},
        "instructions": {"text_instructions": [], "example_question_sqls": []},
    }
    print(f"  Creating Genie space '{title}' with {len(tables)} table(s) …")
    space = w.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=json.dumps(serialized_space),
        description=description,
        title=title,
    )
    return space.space_id


def choose_action(args) -> str:
    """Decide reuse-vs-create. Flags short-circuit; otherwise prompt interactively."""
    if args.space_id:
        return "attach"
    if args.title:
        return "title"
    while True:
        print("\nGenie space setup:")
        print("  1. Attach an existing space by ID")
        print("  2. Reuse or create a space by title")
        choice = input("Choose an option [1/2]: ").strip()
        if choice == "1":
            args.space_id = prompt_value("Genie space ID", None)
            return "attach"
        if choice == "2":
            args.title = prompt_value("Space title", None)
            return "title"
        print("Enter 1 or 2.")


def resolve_space_id(w, args) -> str:
    action = choose_action(args)

    if action == "attach":
        # Validate it exists / is readable.
        space = w.genie.get_space(space_id=args.space_id)
        print(f"  Attaching existing space: {space.title} ({args.space_id})")
        return args.space_id

    existing = find_space_by_title(w, args.title)
    if existing:
        print(f"  Reusing space by title: {existing.title} ({existing.space_id})")
        return existing.space_id
    return create_space(w, args)


def main() -> None:
    args = parse_args()
    ensure_env_file()

    profile = resolve_profile(args.profile)
    if not validate_profile(profile):
        sys.exit(
            f"Profile '{profile}' is not authenticated. Run:\n"
            f"  databricks auth login --profile {profile}"
        )

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=profile)
    space_id = resolve_space_id(w, args)

    print("Writing GENIE_SPACE_ID to .env, app/app.yaml, databricks.yml …")
    update_env_file("GENIE_SPACE_ID", space_id)
    set_app_yaml_env({"GENIE_SPACE_ID": space_id})
    set_bundle_variable_default("genie_space_id", space_id)

    print(f"\nDone. GENIE_SPACE_ID = {space_id}")


if __name__ == "__main__":
    main()
