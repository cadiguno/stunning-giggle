#!/usr/bin/env python3
"""Cluster Steam games by tag profiles, weighted by estimated units sold.

Pipeline:
  1. Parse owners_estimate buckets ("20,000 .. 50,000") into point estimates
     (geometric mean of bucket bounds) used as the units-sold proxy.
  2. Build a game x tag matrix. Tag order in the CSV reflects Steam's vote
     ranking, so earlier tags get higher weight (1/sqrt(rank)). Tags are then
     IDF-weighted (downweights near-universal tags like "Indie") and each
     game's vector is L2-normalized, making KMeans approximate cosine
     (spherical) clustering.
  3. Run KMeans with per-game sample weights derived from units sold, so
     clusters form around where the commercial volume is. Raw units span
     ~6K to ~70M and would let a handful of blockbusters dominate, so the
     default weighting is sqrt(units); use --weight to change.
  4. If --k is not given, sweep a range of k and pick the best cosine
     silhouette score.

Outputs (written next to the input CSV):
  game_clusters.csv    one row per game: cluster id, units estimate, tags
  cluster_summary.csv  one row per cluster: size, units share, distinctive
                       tags (by lift vs. overall prevalence), top games
  cluster_map.png      2D SVD projection colored by cluster, sized by units
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def parse_owners_bucket(s: str) -> float:
    """Geometric mean of an owners bucket; a 0 lower bound becomes upper/10."""
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", s)]
    if len(nums) != 2:
        return np.nan
    lo, hi = nums
    if lo == 0:
        lo = hi / 10
    return float(np.sqrt(lo * hi))


def build_tag_matrix(tag_lists, min_tag_games: int):
    """Game x tag sparse matrix with 1/sqrt(rank) tag weights, then IDF + L2."""
    vocab = {}
    rows, cols, vals = [], [], []
    for i, tags in enumerate(tag_lists):
        for rank, tag in enumerate(tags, start=1):
            j = vocab.setdefault(tag, len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(1.0 / np.sqrt(rank))
    X = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(tag_lists), len(vocab))
    )
    tag_names = np.array(sorted(vocab, key=vocab.get))

    doc_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = doc_freq >= min_tag_games
    X = X[:, keep]
    tag_names = tag_names[keep]
    doc_freq = doc_freq[keep]

    idf = np.log(X.shape[0] / doc_freq)
    X = X.multiply(idf).tocsr()
    return normalize(X), tag_names


def pick_k(X, sample_weight, k_range, seed):
    best_k, best_score = None, -1.0
    # sample the silhouette above 10k games: the full pairwise-distance
    # matrix is O(n^2) memory
    sample = min(X.shape[0], 8000)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed)
        labels = km.fit_predict(X, sample_weight=sample_weight)
        score = silhouette_score(X, labels, metric="cosine",
                                 sample_size=sample, random_state=seed)
        print(f"  k={k:>2}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Selected k={best_k} (silhouette={best_score:.4f})")
    return best_k


def distinctive_tags(X, tag_names, labels, units, cluster, top_n=12):
    """Tags over-represented in a cluster: units-weighted prevalence lift."""
    present = X > 0
    w = units / units.sum()
    overall = present.T.dot(w)
    mask = labels == cluster
    w_c = units[mask] / units[mask].sum()
    prev = present[mask].T.dot(w_c)
    lift = prev / (overall + 1e-9)
    # require the tag on a meaningful share of the cluster's volume
    candidates = np.where(prev >= 0.25)[0]
    ranked = candidates[np.argsort(-lift[candidates])][:top_n]
    return [f"{tag_names[j]} ({prev[j]:.0%})" for j in ranked]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", default="steam_games.csv")
    ap.add_argument("--min-units", type=float, default=10_000,
                    help="drop games below this units estimate (default 10000)")
    ap.add_argument("--min-tag-games", type=int, default=20,
                    help="drop tags present in fewer games than this")
    ap.add_argument("--k", type=int, default=None,
                    help="number of clusters; omit to auto-select")
    ap.add_argument("--k-min", type=int, default=6)
    ap.add_argument("--k-max", type=int, default=18)
    ap.add_argument("--weight", choices=["units", "sqrt", "log", "none"],
                    default="sqrt", help="sample-weight transform of units")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: next to the input CSV)")
    args = ap.parse_args()

    path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    df = df[df["tags"].notna()].copy()

    # tags: JSON array (GDCo export) or semicolon-separated (SteamSpy scrape)
    if df["tags"].iloc[0].strip().startswith("["):
        df["tag_list"] = df["tags"].map(json.loads)
    else:
        df["tag_list"] = df["tags"].str.split(";")
    df = df[df["tag_list"].map(len) > 0]

    # units: actual copies sold (GDCo) or owners-bucket estimate (SteamSpy)
    if "copies_sold" in df.columns:
        df["units_est"] = df["copies_sold"].astype(float)
    else:
        df = df[df["owners_estimate"].notna()]
        df["units_est"] = df["owners_estimate"].map(parse_owners_bucket)
    df = df[df["units_est"] >= args.min_units].reset_index(drop=True)
    print(f"{len(df)} games after min-units filter (>= {args.min_units:,.0f})")

    tag_lists = df["tag_list"].tolist()
    X, tag_names = build_tag_matrix(tag_lists, args.min_tag_games)
    print(f"Tag matrix: {X.shape[0]} games x {X.shape[1]} tags")

    units = df["units_est"].to_numpy()
    sample_weight = {
        "units": units,
        "sqrt": np.sqrt(units),
        "log": np.log10(units),
        "none": None,
    }[args.weight]

    k = args.k or pick_k(X, sample_weight,
                         range(args.k_min, args.k_max + 1), args.seed)
    km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
    df["cluster"] = km.fit_predict(X, sample_weight=sample_weight)

    # per-game output
    df["units_est"] = df["units_est"].round().astype(int)
    game_cols = [c for c in
                 ["appid", "gdco_id", "name", "cluster", "units_est",
                  "release_date", "developers", "publishers", "price_usd",
                  "revenue", "review_count", "review_percent", "positive",
                  "negative", "wishlists", "tags"] if c in df.columns]
    df[game_cols].sort_values(["cluster", "units_est"], ascending=[True, False]) \
        .to_csv(out_dir / "game_clusters.csv", index=False)

    # per-cluster summary
    total_units = units.sum()
    rows = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        top_games = sub.nlargest(5, "units_est")["name"].tolist()
        if "price_usd" in sub.columns:
            med_price = sub["price_usd"].median()
        elif "revenue" in sub.columns:
            # effective gross price per copy actually paid
            med_price = (sub["revenue"] / sub["units_est"]).median().round(2)
        else:
            med_price = np.nan
        rows.append({
            "cluster": c,
            "n_games": len(sub),
            "total_units_est": int(sub["units_est"].sum()),
            "units_share": sub["units_est"].sum() / total_units,
            "median_units": int(sub["units_est"].median()),
            "median_price_usd": med_price,
            **({"total_revenue": int(sub["revenue"].sum())}
               if "revenue" in sub.columns else {}),
            "distinctive_tags": "; ".join(
                distinctive_tags(X, tag_names, df["cluster"].to_numpy(),
                                 units, c)),
            "top_games_by_units": "; ".join(top_games),
        })
    summary = pd.DataFrame(rows).sort_values("total_units_est",
                                             ascending=False)
    summary["units_share"] = summary["units_share"].round(4)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)

    # 2D map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy = TruncatedSVD(n_components=2, random_state=args.seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20")
    sizes = 4 + 40 * (np.log10(units) - np.log10(units).min()) \
        / (np.log10(units).max() - np.log10(units).min())
    for c in range(k):
        m = df["cluster"] == c
        ax.scatter(xy[m, 0], xy[m, 1], s=sizes[m], color=cmap(c % 20),
                   alpha=0.5, label=f"C{c} (n={m.sum()})", linewidths=0)
    ax.legend(fontsize=7, markerscale=2, ncol=2)
    ax.set_title(f"Steam games by tag profile — {k} clusters "
                 f"(weight={args.weight})")
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_map.png", dpi=150)

    print(f"\n{'cl':>3} {'games':>6} {'units share':>12}  distinctive tags")
    for _, r in summary.iterrows():
        tags_short = ", ".join(
            t.split(" (")[0] for t in r["distinctive_tags"].split("; ")[:6])
        print(f"{r['cluster']:>3} {r['n_games']:>6} {r['units_share']:>11.1%}"
              f"  {tags_short}")
    print("\nWrote game_clusters.csv, cluster_summary.csv, cluster_map.png")


if __name__ == "__main__":
    main()
