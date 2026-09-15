from __future__ import annotations

import logging
import os
import platform
import sqlite3
import warnings

##### CONSTANTS ##### (configurable)
CACHE_BUSY_TIMEOUT_MS = 60000
CACHE_DISABLING_VALUES = ("", "none", "off")
APP_NAME = "mozzarellm"


def default_cache_dir(app_name: str) -> str:
    system = platform.system()

    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return os.path.join(base, app_name)

    if system == "Darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", app_name)

    base = os.environ.get("XDG_CACHE_HOME")  # linux
    if base:
        return os.path.join(base, app_name)
    return os.path.join(os.path.expanduser("~"), ".cache", app_name)


def warn_cache_disabled(label: str, cache_path: str | None, env_var: str, error: Exception) -> None:
    """Say loudly, once per failure, that the cache is out of the loop.

    A cache that cannot be read must not become thousands of empty annotations:
    the lookups fall through to the API instead, and the operator is told why.
    """
    message = (
        f"mozzarellm {label} cache at {cache_path} is unusable ({type(error).__name__}: {error}); "
        f"continuing without a cache -- lookups go to the {label} API. Delete the file to rebuild "
        f"it, or set {env_var} to a path on node-local storage."
    )
    logging.error(message)
    warnings.warn(message, stacklevel=3)


def resolve_cache_path(
    cache_path: str | os.PathLike[str] | None,
    *,
    label: str,
    env_var: str,
    default_filename: str,
) -> str | None:
    """Resolve the cache file: explicit argument, then the env var, then the default.

    Setting the env var to an empty value (or "none"/"off") turns the cache off
    outright; pointing it at node-local storage is what keeps concurrent cluster
    jobs off a shared network filesystem. A cache directory that cannot be created
    degrades to "no cache" rather than failing the run.
    """
    if cache_path is None:
        env_value = os.environ.get(env_var)
        if env_value is not None:
            if env_value.strip().lower() in CACHE_DISABLING_VALUES:
                return None
            cache_path = env_value.strip()
        else:
            cache_path = os.path.join(default_cache_dir(APP_NAME), default_filename)
    resolved = os.fspath(cache_path)
    parent = os.path.dirname(resolved)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            warn_cache_disabled(label, resolved, env_var, e)
            return None
    return resolved


def init_cache(
    cache_path: str,
    *,
    label: str,
    env_var: str,
    table_sql: str,
) -> sqlite3.Connection | None:
    """Open the on-disk cache, or return None when it cannot be used safely.

    The journal is the rollback journal, not WAL: WAL needs shared memory between
    the processes on one host and is documented as unsafe over a network
    filesystem, which is how concurrent annotation jobs corrupt a cache that lives
    on NFS. Setting the mode here also converts a cache left in WAL by an earlier
    version. An unwritable file, a failed open, or a database that fails its
    integrity check degrades to "no cache" instead of failing every lookup.

    Note: Uses Python's built-in sqlite3 module (SQLite 3.x).
    Developed with SQLite 3.37+. PRAGMA statements may change in future releases.
    """
    try:
        conn = sqlite3.connect(
            cache_path, timeout=CACHE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(f"PRAGMA busy_timeout={CACHE_BUSY_TIMEOUT_MS}")
        check = conn.execute("PRAGMA quick_check(1)").fetchone()
        if not check or str(check[0]).lower() != "ok":
            raise sqlite3.DatabaseError(
                f"integrity check failed: {check[0] if check else 'no result'}"
            )
        conn.execute(table_sql)
        return conn
    except (sqlite3.Error, OSError) as e:
        warn_cache_disabled(label, cache_path, env_var, e)
        return None
