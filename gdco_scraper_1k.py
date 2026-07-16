#!/usr/bin/env python3
"""Lite GDCo scrape: Steam games with 1,000+ copies sold, core fields only.

Fully separate outputs from the main 10k+ scrape: its own checkpoint DB
(gdco_1k.sqlite) and its own CSV (exports/steam_games_1k.csv). Fetches only
the fields the clustering needs (identity, tags, lifetime totals) — no
quarterly or pre-release history, which is what made the original scrape
slow. Resumable: the pagination cursor is checkpointed after every page.

Usage:
  gdco_scraper_1k.py          run/resume the scrape (auto-exports when done)
  gdco_scraper_1k.py status   show progress
  gdco_scraper_1k.py export   write the CSV from whatever is scraped so far
"""

import csv
import json
import sqlite3
import sys
import time
from datetime import date, datetime

from gdco_scraper import HERE, gql, log  # reuse auth, budget throttle, retries

DB_PATH = HERE / "gdco_1k.sqlite"
CSV_PATH = HERE / "exports" / "steam_games_1k.csv"
RELEASE_CUTOFF = "2021-07-09"  # same window start as the 10k+ dataset
MIN_COPIES = 1_000
PAGE_SIZE = 50


def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS steam_games_1k (
            appid INTEGER PRIMARY KEY,
            name TEXT, release_date TEXT,
            developers TEXT, publishers TEXT, tags TEXT,
            copies_sold INTEGER, revenue INTEGER,
            review_count INTEGER, review_percent INTEGER,
            wishlists INTEGER, fetched_at TEXT
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
        {{ steamApps(first: {PAGE_SIZE}{after},
            filter: {{releaseDate: {{gte: "{RELEASE_CUTOFF}", lte: "{date.today()}"}},
                      copiesSold: {{gte: {MIN_COPIES}}}}},
            orderBy: {{appid: Asc}}) {{
          totalCount
          pageInfo {{ hasNextPage endCursor }}
          edges {{ node {{
            appid name releaseDate copiesSold revenue reviewCount reviewPercent wishlists
            developers {{ steamPartner {{ name }} }}
            publishers {{ steamPartner {{ name }} }}
            tags {{ rank tag {{ name }} }}
          }} }}
        }} }}"""
        data = gql(query)["steamApps"]
        for edge in data["edges"]:
            n = edge["node"]
            con.execute(
                "INSERT OR REPLACE INTO steam_games_1k "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    n["appid"], n["name"], n.get("releaseDate") or "",
                    "; ".join(p["steamPartner"]["name"]
                              for p in n["developers"] if p.get("steamPartner")),
                    "; ".join(p["steamPartner"]["name"]
                              for p in n["publishers"] if p.get("steamPartner")),
                    json.dumps([t["tag"]["name"]
                                for t in sorted(n["tags"],
                                                key=lambda t: t["rank"] or 99)
                                if t.get("tag")]),
                    n["copiesSold"], n["revenue"], n["reviewCount"],
                    n["reviewPercent"], n["wishlists"],
                    datetime.now().isoformat(),
                ),
            )
        con.commit()
        done = con.execute(
            "SELECT COUNT(*) FROM steam_games_1k").fetchone()[0]
        log(f"1k list: {done}/{data['totalCount']} games")
        if not data["pageInfo"]["hasNextPage"]:
            set_state(con, "list_done", "1")
            log("1k list: DONE")
            return
        cursor = data["pageInfo"]["endCursor"]
        set_state(con, "cursor", cursor)
        time.sleep(1)


def export(con):
    rows = con.execute(
        "SELECT appid, name, release_date, developers, publishers, tags, "
        "copies_sold, revenue, review_count, review_percent, wishlists, "
        "fetched_at FROM steam_games_1k ORDER BY appid").fetchall()
    CSV_PATH.parent.mkdir(exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["appid", "name", "release_date", "developers",
                    "publishers", "tags", "copies_sold", "revenue",
                    "review_count", "review_percent", "wishlists",
                    "fetched_at"])
        w.writerows(rows)
    log(f"exported {len(rows)} rows -> {CSV_PATH}")


def status(con):
    done = con.execute("SELECT COUNT(*) FROM steam_games_1k").fetchone()[0]
    finished = bool(get_state(con, "list_done"))
    print(f"games scraped: {done}  complete: {finished}")


def main():
    con = db_connect()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "status":
        status(con)
    elif cmd == "export":
        export(con)
    else:
        log(f"lite scrape: releases since {RELEASE_CUTOFF}, "
            f"min {MIN_COPIES:,} copies, core fields only")
        scrape(con)
        export(con)


if __name__ == "__main__":
    main()
