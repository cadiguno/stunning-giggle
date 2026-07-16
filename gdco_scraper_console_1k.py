#!/usr/bin/env python3
"""Lite GDCo scrape: console games with 1,000+ estimated players, core only.

Console counterpart of gdco_scraper_1k.py: separate outputs (checkpoint DB
gdco_console_1k.sqlite, CSV exports/console_games_1k.csv), core fields plus
the per-platform breakdown, but no quarterly metrics history — that is what
made the original console scrape slow. Column layout matches
console_games.csv so downstream scripts work unchanged. Resumable.

Usage:
  gdco_scraper_console_1k.py          run/resume (auto-exports when done)
  gdco_scraper_console_1k.py status   show progress
  gdco_scraper_console_1k.py export   write the CSV from whatever is scraped
"""

import csv
import json
import sqlite3
import sys
import time
from datetime import date, datetime

from gdco_scraper import HERE, gql, log  # reuse auth, budget throttle, retries

DB_PATH = HERE / "gdco_console_1k.sqlite"
CSV_PATH = HERE / "exports" / "console_games_1k.csv"
RELEASE_CUTOFF = "2021-07-09"  # same window start as the other datasets
MIN_PLAYERS = 1_000
PAGE_SIZE = 25


def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS console_games_1k (
            gdco_id INTEGER PRIMARY KEY,
            name TEXT, developer TEXT, publisher TEXT, release_date TEXT,
            tags TEXT, platforms TEXT,
            player_estimate INTEGER, revenue_gross INTEGER,
            price_usd INTEGER, fetched_at TEXT
        );
        """
    )
    return con


def get_state(con, key, default=None):
    row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con, key, value):
    con.execute(
        "INSERT INTO state VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )
    con.commit()


def scrape(con):
    if get_state(con, "list_done"):
        log("scrape already complete")
        return
    cursor = get_state(con, "cursor")
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""
        {{ gdcoApps(first: {PAGE_SIZE}{after},
            filter: {{releaseDate: {{gte: "{RELEASE_CUTOFF}", lte: "{date.today()}"}},
                      playerEstimate5d: {{gte: {MIN_PLAYERS}}},
                      OR: [{{isXbox: {{equals: true}}}},
                           {{isPlayStation: {{equals: true}}}},
                           {{isSwitch: {{equals: true}}}}]}},
            orderBy: {{gdcoId: Asc}}) {{
          totalCount
          pageInfo {{ hasNextPage endCursor }}
          edges {{ node {{
            gdcoId name developer publisher releaseDate tagNames
            playerEstimate revenueEstimateGross priceUsd
            appPlatforms(first: 10) {{ edges {{ node {{
              platform {{ name }} isConsole releaseDate ratingPercent
              playerEstimate revenueEstimateGross
            }} }} }}
          }} }}
        }} }}"""
        data = gql(query)["gdcoApps"]
        for edge in data["edges"]:
            n = edge["node"]
            platforms = [
                {
                    "platform": (p["node"].get("platform") or {})
                    .get("name") or "?",
                    "is_console": p["node"].get("isConsole"),
                    "release_date": p["node"].get("releaseDate"),
                    "rating_percent": p["node"].get("ratingPercent"),
                    "player_estimate": p["node"].get("playerEstimate"),
                    "revenue_gross": p["node"].get("revenueEstimateGross"),
                }
                for p in n["appPlatforms"]["edges"]
            ]
            con.execute(
                "INSERT OR REPLACE INTO console_games_1k "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    n["gdcoId"], n["name"], n.get("developer"),
                    n.get("publisher"), n.get("releaseDate"),
                    n.get("tagNames"), json.dumps(platforms),
                    n.get("playerEstimate"), n.get("revenueEstimateGross"),
                    n.get("priceUsd"), datetime.now().isoformat(),
                ),
            )
        con.commit()
        done = con.execute(
            "SELECT COUNT(*) FROM console_games_1k").fetchone()[0]
        log(f"console 1k list: {done}/{data['totalCount']} games")
        if not data["pageInfo"]["hasNextPage"]:
            set_state(con, "list_done", "1")
            log("console 1k list: DONE")
            return
        cursor = data["pageInfo"]["endCursor"]
        set_state(con, "cursor", cursor)
        time.sleep(1)


def export(con):
    rows = con.execute(
        "SELECT gdco_id, name, developer, publisher, release_date, tags, "
        "platforms, player_estimate, revenue_gross, price_usd, fetched_at "
        "FROM console_games_1k ORDER BY gdco_id").fetchall()
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gdco_id", "name", "developer", "publisher",
                    "release_date", "tags", "platforms", "player_estimate",
                    "revenue_gross", "price_usd", "fetched_at"])
        w.writerows(rows)
    log(f"exported {len(rows)} rows -> {CSV_PATH}")


def main():
    con = db_connect()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        done = con.execute(
            "SELECT COUNT(*) FROM console_games_1k").fetchone()[0]
        print(f"games scraped: {done}  "
              f"complete: {bool(get_state(con, 'list_done'))}")
    elif cmd == "export":
        export(con)
    else:
        log(f"console lite scrape: releases since {RELEASE_CUTOFF}, "
            f"min {MIN_PLAYERS:,} players, core fields only")
        scrape(con)
        export(con)


if __name__ == "__main__":
    main()
