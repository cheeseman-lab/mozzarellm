#!/usr/bin/env python3
"""UniProt SQLite cache management CLI.

Usage:
    python sqlite_cache.py --stat      Show cache statistics and sample URLs
    python sqlite_cache.py --version    Show SQLite version and WAL mode support
    python sqlite_cache.py --evict <TTL_SECONDS>    Delete entries older than a given TTL (seconds)
    python sqlite_cache.py --clear      Delete ALL cache entries (irreversible)
"""

import argparse
import os
import platform
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
ORANGE = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


def get_cache_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "mozzarellm" / "uniprot_cache.sqlite3"
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "mozzarellm" / "uniprot_cache.sqlite3"
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "mozzarellm" / "uniprot_cache.sqlite3"
    return Path.home() / ".cache" / "mozzarellm" / "uniprot_cache.sqlite3"


def cmd_stats() -> None:
    cache_path = get_cache_path()

    if not cache_path.exists():
        print(f"{RED}Cache not found at: {cache_path}{RESET}")
        print(f"{BLUE}Run some UniProt queries first to populate the cache.{RESET}")
        return

    print(f"{GREEN}Cache found at: {cache_path}{RESET}")
    print(f"Size: {cache_path.stat().st_size / 1024:.2f} KB\n")

    conn = sqlite3.connect(str(cache_path))

    # Total entries
    total = conn.execute("SELECT COUNT(*) FROM uniprot_http_cache").fetchone()[0]
    print(f"{BLUE}Total cached entries: {total}{RESET}")

    if total == 0:
        print(f"{RED}   Cache is empty.{RESET}")
        conn.close()
        return

    # Oldest and newest entries
    oldest = conn.execute("SELECT MIN(created_at) FROM uniprot_http_cache").fetchone()[0]
    newest = conn.execute("SELECT MAX(created_at) FROM uniprot_http_cache").fetchone()[0]

    print(f"   Oldest entry: {datetime.fromtimestamp(oldest).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Newest entry: {datetime.fromtimestamp(newest).strftime('%Y-%m-%d %H:%M:%S')}")

    # Sample URLs
    print(f"\n{BLUE}Sample cached URLs:{RESET}")
    samples = conn.execute("SELECT url, params_json FROM uniprot_http_cache LIMIT 5").fetchall()

    for url, params in samples:
        print(f"-- {url}")
        if params and params != "{}":
            print(f"    Params: {params[:120]}")

    avg_size = conn.execute("SELECT AVG(LENGTH(response_json)) FROM uniprot_http_cache").fetchone()[
        0
    ]
    print(f"\n{MAGENTA}Average response size: {avg_size:.0f} bytes{RESET}")
    conn.close()


def cmd_version() -> None:
    print(f"Python:  {sys.version.split()[0]}")
    print(f"SQLite:  {sqlite3.sqlite_version}")

    tmp_db = tempfile.mktemp(suffix=".db")
    try:
        conn = sqlite3.connect(tmp_db)
        wal_mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        conn.close()
    finally:
        if os.path.exists(tmp_db):
            os.remove(tmp_db)

    if wal_mode == "wal":
        print(f"{GREEN} WAL mode is supported{RESET}")
    else:
        print(f"{ORANGE} WAL mode not supported (got: {wal_mode}).{RESET}")

    # Parse version to check minimum requirement
    version_parts = [int(x) for x in sqlite3.sqlite_version.split(".")]
    major, minor = version_parts[0], version_parts[1]

    if major > 3 or (major == 3 and minor >= 7):
        print(f"{GREEN}SQLite {sqlite3.sqlite_version} meets minimum requirement (3.7.0+){RESET}")
    else:
        print(f"{RED}SQLite {sqlite3.sqlite_version} is below minimum requirement (3.7.0+){RESET}")


def cmd_evict(ttl_seconds: int) -> None:
    cache_path = get_cache_path()
    if not cache_path.exists():
        print(f"{RED}Cache not found at: {cache_path}{RESET}")
        return
    confirm = (
        input(f"Evict entries older than {ttl_seconds}s from {cache_path}? [y/N] ").strip().lower()
    )
    if confirm != "y":
        print("Aborted.")
        return
    conn = sqlite3.connect(str(cache_path))
    cutoff = int(time.time()) - ttl_seconds
    cursor = conn.execute("DELETE FROM uniprot_http_cache WHERE created_at < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.close()
    print(f"{GREEN}Evicted {deleted} expired entries (TTL: {ttl_seconds}s){RESET}")


def cmd_clear() -> None:
    cache_path = get_cache_path()
    if not cache_path.exists():
        print(f"{RED}Cache not found at: {cache_path}{RESET}")
        return
    confirm = (
        input(f"Delete ALL entries from {cache_path}? WARNING: This is irreversible! [y/N] ")
        .strip()
        .lower()
    )
    if confirm != "y":
        print("Aborted.")
        return
    conn = sqlite3.connect(str(cache_path))
    cursor = conn.execute("DELETE FROM uniprot_http_cache")
    deleted = cursor.rowcount
    conn.close()
    print(f"{GREEN}Cleared {deleted} entries from cache.{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UniProt SQLite cache management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stat", action="store_true", help="Show cache statistics")
    group.add_argument("--version", action="store_true", help="Show SQLite version info")
    group.add_argument(
        "--evict",
        metavar="TTL_SECONDS",
        type=int,
        help="Delete entries older than TTL_SECONDS",
    )
    group.add_argument("--clear", action="store_true", help="Delete ALL cache entries")

    args = parser.parse_args()

    if args.stat:
        cmd_stats()
    elif args.version:
        cmd_version()
    elif args.evict is not None:
        cmd_evict(args.evict)
    elif args.clear:
        cmd_clear()


if __name__ == "__main__":
    main()
