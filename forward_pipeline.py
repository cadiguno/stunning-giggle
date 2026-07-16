#!/usr/bin/env python3
"""Forward-looking pipeline ranking for PQube: aggregate current wishlist
demand on UNRELEASED games (next 180 days, matching the validated backtest
window) by tag cluster.

Pulls upcoming releases via the GDCo GraphQL steamApps query (same shape as
scrape_all_releases.py), assigns each to a tag cluster with the centroid
method from assign_console_clusters.py, then ranks clusters by pipeline
demand, demand-per-title, and demand/supply whitespace, alongside each
cluster's historical hit production from the hype backtest panel.

Writes exports/forward_pipeline_games.csv, exports/forward_pipeline_ranking.csv.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from cluster_games_by_tags import build_tag_matrix
from gdco_scraper import gql
from genre_lexicon import LABELS

HERE = Path(__file__).resolve().parent
LOOKAHEAD = 180
PAGE_SIZE = 100


def page_query(cursor, page_size, gte, lte):
    after = f', after: "{cursor}"' if cursor else ""
    return f"""
    {{ steamApps(first: {page_size}{after},
        filter: {{releaseDate: {{gte: "{gte}", lte: "{lte}"}}}},
        orderBy: {{appid: Asc}}) {{
      totalCount
      pageInfo {{ hasNextPage endCursor }}
      edges {{ node {{
        appid name releaseDate wishlists
        tags {{ rank tag {{ name }} }}
      }} }}
    }} }}"""


def fetch_upcoming():
    gte = (date.today() + timedelta(days=1)).isoformat()
    lte = (date.today() + timedelta(days=LOOKAHEAD)).isoformat()
    out, cursor, page_size = [], None, PAGE_SIZE
    while True:
        try:
            data = gql(page_query(cursor, page_size, gte, lte))["steamApps"]
        except RuntimeError as exc:
            if page_size > 20:
                page_size //= 2
                print(f"query failed ({exc}); page size -> {page_size}")
                time.sleep(10)
                continue
            raise
        for e in data["edges"]:
            n = e["node"]
            tags = [t["tag"]["name"] for t in
                    sorted(n["tags"], key=lambda t: t["rank"] or 99)
                    if t.get("tag")]
            out.append({"appid": n["appid"], "name": n["name"],
                        "release_date": n.get("releaseDate") or "",
                        "wishlists": n.get("wishlists"),
                        "tags": json.dumps(tags)})
        if len(out) % 1000 < page_size:
            print(f"fetched {len(out)} / {data['totalCount']}", flush=True)
        if not data["pageInfo"]["hasNextPage"]:
            break
        cursor = data["pageInfo"]["endCursor"]
        time.sleep(0.8)
    print(f"upcoming games ({gte}..{lte}): {len(out)}")
    return pd.DataFrame(out)


def assign_clusters(up):
    steam = pd.read_csv(HERE / "exports" / "game_clusters.csv")
    steam_tags = steam["tags"].map(
        lambda s: json.loads(s) if s.startswith("[") else s.split(";")
    ).tolist()
    X, tag_names = build_tag_matrix(steam_tags, 20)
    vocab = {t: j for j, t in enumerate(tag_names)}
    labels = steam["cluster"].to_numpy()
    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(X.shape[0] / doc_freq)
    centroids = normalize(np.vstack([
        np.asarray(X[labels == c].mean(axis=0)).ravel()
        for c in range(labels.max() + 1)]))

    rows, cols, vals, keep = [], [], [], []
    for i, tags_json in enumerate(up["tags"]):
        tags = json.loads(tags_json)
        known = [(r, t) for r, t in enumerate(tags, start=1) if t in vocab]
        if len(known) < 3:
            continue
        keep.append(i)
        for rank, tag in known:
            rows.append(len(keep) - 1)
            cols.append(vocab[tag])
            vals.append(1.0 / np.sqrt(rank) * idf[vocab[tag]])
    Xu = normalize(sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(keep), len(tag_names))))
    up = up.iloc[keep].reset_index(drop=True)
    sims = Xu @ centroids.T
    up["cluster"] = np.asarray(sims.argmax(axis=1)).ravel()
    up["cluster_sim"] = np.asarray(sims.max(axis=1)).ravel().round(3)
    return up


def main():
    up = fetch_upcoming()
    up["wishlists"] = pd.to_numeric(up["wishlists"], errors="coerce")
    print(f"wishlists coverage: {up['wishlists'].notna().mean()*100:.0f}% "
          f"non-null, median {up['wishlists'].median():.0f}")
    up = assign_clusters(up)
    up.to_csv(HERE / "exports" / "forward_pipeline_games.csv", index=False)

    up["wl"] = up["wishlists"].fillna(0)
    agg = up.groupby("cluster").agg(
        n_upcoming=("appid", "size"),
        wishlists=("wl", "sum"),
        wl_per_title=("wl", "mean"),
        wl_median=("wl", "median")).round(0)
    agg["demand_share"] = (agg["wishlists"] / agg["wishlists"].sum() * 100)
    agg["supply_share"] = (agg["n_upcoming"] / agg["n_upcoming"].sum() * 100)
    agg["whitespace"] = (agg["demand_share"] / agg["supply_share"])

    # historical context from the validated backtest panel
    panel = pd.read_csv(HERE / "exports" / "hype_backtest_panel.csv")
    hist = panel.groupby("cluster").agg(
        hist_hits_per_yr=("n_hits", "mean"),
        hist_followers=("followers", "mean"))
    agg = agg.join(hist)
    agg["hits_per_100_titles"] = (agg["hist_hits_per_yr"]
                                  / agg["n_upcoming"] * 100)
    agg["label"] = [LABELS.get(c, "?") for c in agg.index]

    top3 = (up.sort_values("wl", ascending=False).groupby("cluster")
            .head(3).groupby("cluster")["name"]
            .apply(lambda s: " | ".join(s.astype(str))))
    agg["top_upcoming"] = top3

    agg = agg.sort_values("whitespace", ascending=False)
    cols = ["label", "n_upcoming", "wishlists", "wl_per_title",
            "demand_share", "supply_share", "whitespace",
            "hist_hits_per_yr", "top_upcoming"]
    out = agg[cols].round(2)
    out.to_csv(HERE / "exports" / "forward_pipeline_ranking.csv")
    pd.set_option("display.width", 200)
    print("\ncluster pipeline ranking (next 180 days), by whitespace "
          "(demand share / supply share):")
    print(out.drop(columns=["top_upcoming"]).to_string())
    print("\ntop upcoming titles per cluster:")
    for c, r in out.iterrows():
        print(f"  c{c} {r['label']}: {r['top_upcoming']}")


if __name__ == "__main__":
    main()
