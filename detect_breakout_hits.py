#!/usr/bin/env python3
"""Mechanically identify breakout indie hits in the GDCo Steam dataset.

Universe: indie-priced games from exports/game_clusters_under30.csv (release
price <= $29.99 per make_under30.py's estimate; games with no price signal
are kept there, as are PQube titles regardless of price).

Sales velocity is units/day computed from diffs of the cumulative quarterly
snapshots; the release quarter is normalized by days since release, so
partial quarters compare fairly with full ones.

Two triggers:
  - instant:  velocity over the first ~90 days on sale >= INSTANT_UPD
  - breakout: a later quarter jumps to >= JUMP_X times the previous quarter's
    velocity with at least JUMP_MIN_UNITS incremental units (the Among Us /
    Vampire Survivors pattern; a normal sales curve only decays, so any big
    re-acceleration is a real event)

Writes exports/breakout_hits.csv, ranked by peak quarterly velocity.
"""

import json
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INSTANT_UPD = 5_000        # units/day over the first ~90 days (~450k/quarter)
JUMP_X = 4.0               # velocity multiple vs previous quarter
JUMP_MIN_UNITS = 150_000   # incremental units in the jump quarter
EARLY_WINDOW_DAYS = 90

KNOWN_EXAMPLES = {1794680: "Vampire Survivors", 2379780: "Balatro",
                  3164500: "Schedule I", 4704690: "MECCHA CHAMELEON"}


def game_velocities(rows, release):
    """rows: [(snap_date, cumulative_copies)] sorted. Returns list of dicts
    with quarter span, incremental units and units/day."""
    out = []
    prev_date, prev_cum = release, 0
    for snap, cum, quarter in rows:
        if cum is None or pd.isna(cum):
            continue
        snap = date.fromisoformat(snap)
        days = max((snap - prev_date).days, 1)
        inc = max(cum - prev_cum, 0)
        out.append({"quarter": quarter, "end": snap, "days": days,
                    "inc": inc, "upd": inc / days})
        prev_date, prev_cum = snap, cum
    return out


def main():
    universe = pd.read_csv(HERE / "exports" / "game_clusters_under30.csv")
    keep = universe.set_index("appid")[["cluster", "est_list_price"]]

    con = sqlite3.connect(HERE / "gdco_data.sqlite")
    games = pd.read_sql(
        "SELECT appid, name, release_date, developers, publishers, tags, "
        "copies_sold FROM steam_games", con)
    quarterly = pd.read_sql(
        "SELECT appid, quarter, snap_date, copies_sold, revenue "
        "FROM steam_quarterly ORDER BY appid, snap_date", con)
    con.close()

    # publisher scale across the whole 10k+ dataset (before the indie screen):
    # a mechanical flag for AAA/F2P brands that slip through the price proxy
    pub_units = games.groupby("publishers")["copies_sold"].transform("sum")
    games["publisher_other_units"] = pub_units - games["copies_sold"]

    games = games[games["appid"].isin(keep.index)]
    release_by_appid = games.set_index("appid")["release_date"]

    # launch-quarter effective price (revenue/copies at first positive
    # snapshot): ~0 flags F2P, which the price-cap screen keeps by design
    pos = quarterly[(quarterly["copies_sold"] > 0) & (quarterly["revenue"] > 0)]
    eff_price = pos.sort_values("snap_date").groupby("appid").first()
    eff_price = eff_price["revenue"] / eff_price["copies_sold"]

    # drop pre-release snapshots (0-copy placeholders and pre-order rows):
    # they zero out launch velocity and fake huge jumps
    quarterly = quarterly.merge(release_by_appid.rename("release"),
                                left_on="appid", right_index=True)
    quarterly = quarterly[quarterly["snap_date"] >= quarterly["release"]]
    hits = []
    for appid, grp in quarterly.groupby("appid"):
        if appid not in keep.index:
            continue
        meta = games[games["appid"] == appid]
        if meta.empty or not meta.iloc[0]["release_date"]:
            continue
        meta = meta.iloc[0]
        release = date.fromisoformat(meta["release_date"])
        vel = game_velocities(
            list(grp[["snap_date", "copies_sold", "quarter"]]
                 .itertuples(index=False, name=None)), release)
        if not vel:
            continue

        # instant: units/day across snapshots covering the first ~90 days
        span_days = span_units = 0
        for v in vel:
            span_days += v["days"]
            span_units += v["inc"]
            if span_days >= EARLY_WINDOW_DAYS:
                break
        early_upd = span_units / max(span_days, 1)

        trigger, hit = None, None
        for i in range(1, len(vel)):
            prev, cur = vel[i - 1], vel[i]
            # prev must be a real trading period, not a 1-day release sliver
            if (cur["inc"] >= JUMP_MIN_UNITS and prev["upd"] > 0
                    and prev["days"] >= 14
                    and cur["upd"] >= JUMP_X * prev["upd"]):
                trigger, hit = "breakout", (cur, prev)
                break
        if trigger is None and early_upd >= INSTANT_UPD:
            trigger, hit = "instant", (vel[0], None)
        if trigger is None:
            continue

        cur, prev = hit
        tags = json.loads(meta["tags"] or "[]")
        hits.append({
            "appid": appid, "name": meta["name"],
            "release_date": meta["release_date"],
            "developers": meta["developers"], "publishers": meta["publishers"],
            "cluster": keep.loc[appid, "cluster"],
            "est_list_price": keep.loc[appid, "est_list_price"],
            "eff_price_launch": round(eff_price.get(appid, float("nan")), 2),
            "publisher_other_units": meta["publisher_other_units"],
            "trigger": trigger, "hit_quarter": cur["quarter"],
            "hit_units_per_day": round(cur["upd"]),
            "prev_units_per_day": round(prev["upd"]) if prev else None,
            "jump_x": round(cur["upd"] / prev["upd"], 1) if prev else None,
            "hit_quarter_units": cur["inc"],
            "early_units_per_day": round(early_upd),
            "total_copies": meta["copies_sold"],
            "top_tags": "; ".join(tags[:8]),
        })

    df = pd.DataFrame(hits).sort_values("hit_units_per_day", ascending=False)
    out = HERE / "exports" / "breakout_hits.csv"
    df.to_csv(out, index=False)

    print(f"universe: {len(games)} indie-priced games with quarterly data")
    print(f"hits: {len(df)} total — "
          f"{(df['trigger'] == 'instant').sum()} instant, "
          f"{(df['trigger'] == 'breakout').sum()} breakout -> {out.name}")
    for appid, label in KNOWN_EXAMPLES.items():
        row = df[df["appid"] == appid]
        status = (f"{row.iloc[0]['trigger']} in {row.iloc[0]['hit_quarter']}"
                  if not row.empty else "NOT FLAGGED")
        print(f"  check {label}: {status}")
    print("\ntop 25 by hit-quarter velocity:")
    cols = ["name", "release_date", "trigger", "hit_quarter",
            "hit_units_per_day", "jump_x", "total_copies"]
    print(df[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
