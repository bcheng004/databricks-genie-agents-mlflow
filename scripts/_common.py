"""Shared helpers for the setup CLIs: repo paths, .env I/O, and YAML round-trips.

Mirrors the idempotent, formatting-preserving approach from the reference repo:
- .env values are cached so reruns are cheap and non-destructive.
- app/app.yaml and databricks.yml are edited in place with ruamel.yaml so
  comments, quoting, and ordering survive.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
APP_YAML_PATH = REPO_ROOT / "app" / "app.yaml"
DATABRICKS_YML_PATH = REPO_ROOT / "databricks.yml"


# ---------------------------------------------------------------------------
# .env handling (idempotent, position-preserving)
# ---------------------------------------------------------------------------
def ensure_env_file() -> None:
    """Create .env from .env.example (or empty) if it does not exist yet."""
    if ENV_PATH.exists():
        return
    if ENV_EXAMPLE_PATH.exists():
        ENV_PATH.write_text(ENV_EXAMPLE_PATH.read_text())
    else:
        ENV_PATH.write_text("")


def get_env_value(key: str) -> str | None:
    """Return the active (uncommented) value for key from .env, quotes stripped."""
    if not ENV_PATH.exists():
        return None
    active = re.compile(rf"^\s*{re.escape(key)}=(.*)$")
    for line in ENV_PATH.read_text().splitlines():
        m = active.match(line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    return None


def update_env_file(key: str, value: str) -> None:
    """Set key=value in .env.

    If a commented `# KEY=...` placeholder exists, replace it in place so the
    value keeps its documented position; otherwise replace an active line or
    append. Any duplicate active/commented lines for the key are dropped.
    """
    ensure_env_file()
    lines = ENV_PATH.read_text().splitlines()
    active = re.compile(rf"^\s*{re.escape(key)}=.*$")
    commented = re.compile(rf"^#\s*{re.escape(key)}=.*$")
    new_line = f"{key}={value}"

    out: list[str] = []
    placed = False
    for line in lines:
        if active.match(line) or commented.match(line):
            if not placed:
                out.append(new_line)
                placed = True
            # drop any further duplicates
            continue
        out.append(line)
    if not placed:
        out.append(new_line)

    ENV_PATH.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# YAML round-trip helpers
# ---------------------------------------------------------------------------
def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def load_yaml(path: Path):
    y = _yaml()
    with open(path) as f:
        return y, y.load(f)


def dump_yaml(y: YAML, data, path: Path) -> None:
    with open(path, "w") as f:
        y.dump(data, f)


def set_app_yaml_env(updates: dict[str, str]) -> None:
    """Upsert env entries in app/app.yaml (the values the running app reads)."""
    y, data = load_yaml(APP_YAML_PATH)
    env = data.setdefault("env", [])
    by_name = {e.get("name"): e for e in env if isinstance(e, dict)}
    for name, value in updates.items():
        if name in by_name:
            by_name[name]["value"] = DQ(value)
        else:
            env.append({"name": name, "value": DQ(value)})
    dump_yaml(y, data, APP_YAML_PATH)


def set_bundle_variable_default(name: str, value: str) -> None:
    """Update a variables.<name>.default in databricks.yml if that var exists."""
    y, data = load_yaml(DATABRICKS_YML_PATH)
    variables = data.get("variables")
    if isinstance(variables, dict) and name in variables:
        variables[name]["default"] = DQ(value)
        dump_yaml(y, data, DATABRICKS_YML_PATH)


# ---------------------------------------------------------------------------
# Databricks CLI / SDK helpers
# ---------------------------------------------------------------------------
def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    """Run a `databricks` CLI command, returning the completed process."""
    return subprocess.run(
        ["databricks", *args], capture_output=True, text=True, check=False
    )


def validate_profile(profile: str) -> bool:
    """Return True if `databricks current-user me` succeeds for the profile."""
    result = run_cli(["current-user", "me", "--profile", profile])
    return result.returncode == 0


def list_profiles() -> list[str]:
    """Return configured profile names from `databricks auth profiles` (best effort)."""
    result = run_cli(["auth", "profiles"])
    if result.returncode != 0:
        return []
    names: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split()[0]
        if first.lower() == "name":  # skip the table header
            continue
        names.append(first)
    return names


# ---------------------------------------------------------------------------
# Interactive prompts (shared by the setup CLIs)
# ---------------------------------------------------------------------------
def resolve_profile(provided: str | None) -> str:
    """Return the profile to use, prompting interactively when not supplied.

    Priority: explicit value → DATABRICKS_CONFIG_PROFILE env / .env → interactive
    prompt (choose from configured profiles by number or name, or type a new one).
    """
    if provided:
        return provided

    default = os.environ.get("DATABRICKS_CONFIG_PROFILE") or get_env_value(
        "DATABRICKS_CONFIG_PROFILE"
    )
    profiles = list_profiles()

    if profiles:
        print("Available Databricks profiles:")
        for i, name in enumerate(profiles, 1):
            marker = "  (default)" if name == default else ""
            print(f"  {i}. {name}{marker}")
        prompt = "Select a profile [number or name]"
    else:
        prompt = "Enter Databricks profile name"
    if default:
        prompt += f" (default: {default})"
    prompt += ": "

    while True:
        choice = input(prompt).strip()
        if not choice:
            if default:
                return default
            print("A profile is required.")
            continue
        if choice.isdigit() and profiles:
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                return profiles[idx]
            print(f"Enter a number between 1 and {len(profiles)}.")
            continue
        return choice


def prompt_value(
    label: str, provided: str | None, env_key: str | None = None, fallback: str = ""
) -> str:
    """Return a value, prompting interactively when the flag is not supplied.

    Priority: explicit flag → interactive prompt. The prompt's default is the
    cached .env value (env_key) if present, else the given fallback; pressing
    Enter accepts it. A non-empty default is optional — with no default, the
    prompt repeats until a value is entered. An empty string is treated as a
    real value (only None triggers the prompt), so `--flag ""` is respected.
    """
    if provided is not None:
        return provided
    default = (get_env_value(env_key) if env_key else None) or fallback
    suffix = f" (default: {default})" if default else ""
    while True:
        choice = input(f"{label}{suffix}: ").strip()
        if choice:
            return choice
        if default:
            return default
        print(f"{label} is required.")
