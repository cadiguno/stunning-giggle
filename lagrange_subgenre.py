#!/usr/bin/env python3
"""Subgenre-level robustness check of the chatter-leads-hits hypothesis.

The cluster-level backtest was null; this tests whether demand waves exist
at SUBGENRE granularity (survivors-like, extraction, coop-horror, ...) that
the 18 broad clusters wash out.

Stages (resumable, all state in subgenre.sqlite):
  prefilter  regex recall gate over genre_chatter.sqlite -> candidates table
             (subgenre vocab + ~150 exemplar game names); uniform random
             downsample to CAP if needed (sampling fraction stored, shares
             corrected in analyze)
  classify   gpt-4.1-nano via Lagrange, concurrent + checkpointed: each
             candidate -> one subgenre label (or none)
  analyze    monthly demand share per subgenre; event study around known
             subgenre-defining hits with the same growth stat as
             genre_backtest.py -> exports/subgenre_*.csv/png

Usage: lagrange_subgenre.py prefilter|classify|analyze
"""

import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHATTER_DB = HERE / "genre_chatter.sqlite"
DB = HERE / "subgenre.sqlite"
CAP = 130_000
MODEL = "openai/gpt-4.1-nano"
WORKERS = 12
BODY_CHARS = 400
RECENT, BASE_LO, BASE_HI = 6, 7, 18   # same growth windows as genre_backtest

# label: (gloss for the LLM, [prefilter vocab + exemplars], [event months])
TAXONOMY = {
    "survivors_like": ("horde-survival bullet heaven (Vampire Survivors)",
        ["vampire survivors", "brotato", "halls of torment", "holocure",
         "20 minutes till dawn", "bullet heaven", "horde survival",
         "survivors-like", "survivor-like", "survivors like", "megabonk"],
        ["2022-01"]),
    "extraction": ("extraction shooter/RPG (Tarkov, Dark and Darker)",
        ["tarkov", "hunt showdown", "marauders", "dark and darker",
         "extraction shooter", "extraction game", "extraction rpg"],
        ["2023-02"]),
    "coop_horror": ("co-op horror with friends (Phasmophobia, Lethal Company)",
        ["phasmophobia", "lethal company", "content warning", "devour",
         "gtfo", "demonologist", "forewarned", "co-op horror", "coop horror",
         "horror with friends", "horror game with friends"],
        ["2020-09", "2023-10"]),
    "boomer_shooter": ("retro fast FPS (Dusk, Ultrakill)",
        ["\\bdusk\\b", "ultrakill", "amid evil", "ion fury", "prodeus",
         "boomer shooter", "retro fps", "quake-like", "doom clone"],
        ["2020-09"]),
    "cozy_farming": ("cozy farming/life sim (Stardew Valley)",
        ["stardew", "harvest moon", "story of seasons", "coral island",
         "sun haven", "fields of mistria", "dinkum", "farming sim",
         "farm game", "cozy game", "animal crossing", "\\bpalia\\b"],
        []),
    "creature_collector": ("monster taming/collecting (Pokemon-like, Palworld)",
        ["pokemon like", "pokemon-like", "like pokemon", "temtem",
         "cassette beasts", "palworld", "monster taming", "monster tamer",
         "creature collector", "coromon", "nexomon", "monster catching"],
        ["2024-01"]),
    "roguelike_deckbuilder": ("roguelike deckbuilder (Slay the Spire, Balatro)",
        ["slay the spire", "monster train", "balatro", "inscryption",
         "roguelike deckbuilder", "deckbuilder roguelike",
         "deckbuilding roguelike", "card roguelike", "griftlands",
         "wildfrost"],
        ["2024-02"]),
    "autobattler": ("autobattler/auto-chess (TFT, Backpack Battles)",
        ["auto battler", "autobattler", "auto chess", "teamfight tactics",
         "super auto pets", "mechabellum", "backpack battles"],
        ["2024-03"]),
    "colony_sim": ("colony sim (RimWorld, Dwarf Fortress)",
        ["rimworld", "dwarf fortress", "colony sim", "oxygen not included",
         "going medieval", "colony management"],
        []),
    "automation_factory": ("factory/automation (Factorio, Satisfactory)",
        ["factorio", "satisfactory", "dyson sphere", "shapez",
         "automation game", "factory game", "factory builder", "techtonica",
         "captain of industry"],
        ["2021-01"]),
    "survival_craft": ("open-world survival craft (Valheim, Rust)",
        ["valheim", "\\brust\\b", "\\bark\\b", "conan exiles", "subnautica",
         "sons of the forest", "the forest", "grounded", "enshrouded",
         "7 days to die", "v rising", "survival craft", "open world survival",
         "nightingale", "\\bicarus\\b", "smalland", "survival game"],
        ["2021-02", "2024-01"]),
    "soulslike": ("souls-like action RPG (Elden Ring, Dark Souls)",
        ["dark souls", "elden ring", "sekiro", "bloodborne", "lies of p",
         "\\bnioh\\b", "soulslike", "souls-like", "souls like",
         "another crab"],
        ["2022-02"]),
    "metroidvania": ("metroidvania (Hollow Knight)",
        ["hollow knight", "silksong", "metroidvania", "ori and the",
         "blasphemous", "nine sols", "animal well", "lost crown"],
        ["2024-05", "2025-09"]),
    "city_builder": ("city builder (Cities Skylines, Manor Lords)",
        ["cities skylines", "cities: skylines", "city builder", "manor lords",
         "against the storm", "frostpunk", "timberborn", "\\banno\\b",
         "tropico", "city building"],
        ["2024-04"]),
    "job_sim": ("chill job simulator (PowerWash, Supermarket Simulator)",
        ["powerwash", "house flipper", "supermarket simulator",
         "gas station simulator", "pc building simulator", "job simulator",
         "euro truck", "truck simulator", "lawn mowing", "job sim"],
        ["2024-02"]),
    "social_deduction": ("social deduction (Among Us)",
        ["among us", "goose goose duck", "social deduction", "town of salem",
         "project winter", "\\bdeceit\\b"],
        ["2020-08"]),
    "physics_party": ("physics/rage co-op party (Chained Together, PEAK)",
        ["chained together", "gang beasts", "human fall flat", "pico park",
         "getting over it", "golfing over it", "only up", "climbing game",
         "rage game", "bread and fred", "a difficult game about climbing"],
        ["2023-06", "2025-06"]),
    "idle_incremental": ("idle/incremental/clicker",
        ["idle game", "incremental game", "clicker game", "cookie clicker",
         "melvor", "ngu idle", "leaf blower revolution"],
        []),
    "fishing_chill": ("chill fishing (Dredge, Dave the Diver, Webfishing)",
        ["\\bdredge\\b", "dave the diver", "webfishing", "fishing game",
         "moonglow bay", "cat goes fishing"],
        ["2023-03"]),
    "horror_single": ("single-player narrative/survival horror",
        ["resident evil", "silent hill", "outlast", "amnesia", "\\bsoma\\b",
         "\\bvisage\\b", "madison", "mortuary assistant", "signalis",
         "crow country"],
        []),
}


def prefilter():
    con = sqlite3.connect(CHATTER_DB)
    posts = con.execute(
        "SELECT id, month, title, selftext FROM posts").fetchall()
    con.close()
    terms = []
    for _, (_, vocab, _) in TAXONOMY.items():
        terms += [t if t.startswith("\\b") else re.escape(t) for t in vocab]
    rx = re.compile("|".join(terms), re.IGNORECASE)
    cand = [(pid, mo, (ti or "") + "\n" + (bo or "")[:BODY_CHARS])
            for pid, mo, ti, bo in posts
            if rx.search((ti or "") + " " + (bo or ""))]
    frac = 1.0
    if len(cand) > CAP:
        import random
        random.seed(7)
        frac = CAP / len(cand)
        cand = random.sample(cand, CAP)
    out = sqlite3.connect(DB)
    out.execute("CREATE TABLE IF NOT EXISTS candidates "
                "(id TEXT PRIMARY KEY, month TEXT, text TEXT)")
    out.execute("CREATE TABLE IF NOT EXISTS labels "
                "(id TEXT PRIMARY KEY, label TEXT)")
    out.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    out.executemany("INSERT OR IGNORE INTO candidates VALUES (?,?,?)", cand)
    out.execute("INSERT OR REPLACE INTO meta VALUES ('frac', ?)", (str(frac),))
    out.commit()
    print(f"candidates: {len(cand)} of {len(posts)} posts "
          f"(sample frac {frac:.3f}); est ~{len(cand)*450/1e6:.0f}M input tokens")


SYSTEM = ("You label Reddit posts from r/gamingsuggestions. Pick the ONE "
          "subgenre the poster is asking for. Labels:\n"
          + "\n".join(f"{k}: {v[0]}" for k, v in TAXONOMY.items())
          + "\nnone: none of the above / unclear\n"
          "Reply with the label only.")
VALID = set(TAXONOMY) | {"none"}


def classify():
    import openai
    key = [l.split("=", 1)[1].strip() for l in open(HERE / "LagrangeKey.env")
           if l.startswith("LAGRANGE_API_KEY")][0]
    client = openai.OpenAI(
        api_key=key, timeout=60,
        base_url="https://lagrange.uksouth.cloudapp.azure.com/openai")
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    todo = con.execute(
        "SELECT c.id, c.text FROM candidates c "
        "LEFT JOIN labels l ON l.id = c.id WHERE l.id IS NULL").fetchall()
    print(f"to classify: {len(todo)}", flush=True)

    def one(pid, text):
        delay = 2
        for _ in range(8):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=8,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": text[:600]}])
                lab = (r.choices[0].message.content or "").strip().lower()
                return pid, lab if lab in VALID else "none"
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return pid, None

    done = err = 0
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(one, pid, tx) for pid, tx in todo]
        buf = []
        for f in as_completed(futs):
            pid, lab = f.result()
            if lab is None:
                err += 1
                continue
            buf.append((pid, lab))
            done += 1
            if len(buf) >= 200:
                con.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)",
                                buf)
                con.commit()
                buf = []
            if done % 2000 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(todo)} ({rate:.1f}/s, {err} errors)",
                      flush=True)
        if buf:
            con.executemany("INSERT OR REPLACE INTO labels VALUES (?,?)", buf)
            con.commit()
    print(f"classify done: {done} labeled, {err} gave up "
          f"(rerun to retry)", flush=True)


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    con = sqlite3.connect(DB)
    lab = pd.read_sql("SELECT c.month, l.label FROM labels l "
                      "JOIN candidates c ON c.id = l.id", con)
    frac = float(con.execute("SELECT v FROM meta WHERE k='frac'")
                 .fetchone()[0])
    con.close()
    chat = sqlite3.connect(CHATTER_DB)
    totals = pd.read_sql("SELECT month, COUNT(*) n FROM posts GROUP BY month",
                         chat).set_index("month")["n"]
    chat.close()

    months = sorted(totals.index)
    m_ix = {m: i for i, m in enumerate(months)}
    counts = (lab[lab["label"] != "none"]
              .groupby(["month", "label"]).size().unstack(fill_value=0)
              .reindex(months, fill_value=0))
    share = counts.div(frac).add(0.5).div(totals + 1, axis=0)
    S = share.rolling(3, center=True, min_periods=1).mean()

    def growth(sg, ix):
        if ix - BASE_HI < 0:
            return np.nan
        s = S[sg].to_numpy()
        r = s[ix - RECENT:ix].mean()
        b = s[ix - BASE_HI:ix - BASE_LO + 1].mean()
        return float(np.log(r / b)) if r > 0 and b > 0 else np.nan

    print(f"labeled: {len(lab)}; non-none: "
          f"{len(lab[lab['label'] != 'none'])}")
    print(f"label totals:\n{lab['label'].value_counts().to_string()}\n")

    rows = []
    for sg, (_, _, events) in TAXONOMY.items():
        if sg not in S.columns:
            continue
        placebo = [growth(sg, i) for i in range(BASE_HI, len(months))]
        placebo = [g for g in placebo if not np.isnan(g)]
        for ev in events:
            if ev not in m_ix:
                continue
            g = growth(sg, m_ix[ev])
            if np.isnan(g) or not placebo:
                continue
            pct = 100 * float(np.mean([p < g for p in placebo]))
            rows.append({"subgenre": sg, "event": ev, "g": round(g, 3),
                         "own_placebo_pctile": round(pct, 1)})
    ev = pd.DataFrame(rows)
    print("pre-event demand growth vs subgenre's own history:")
    print(ev.to_string(index=False))
    med = ev["own_placebo_pctile"].median() if len(ev) else float("nan")
    print(f"\nmedian percentile across {len(ev)} events: {med:.0f} "
          "(>>50 would mean chatter rises before subgenre-defining hits)")
    ev.to_csv(HERE / "exports" / "subgenre_event_study.csv", index=False)

    n = len(S.columns)
    nrow = (n + 2) // 3
    fig, axes = plt.subplots(nrow, 3, figsize=(15, 2.2 * nrow), sharex=True)
    x = pd.PeriodIndex(months, freq="M").to_timestamp()
    for ax, sg in zip(axes.flat, S.columns):
        ax.plot(x, S[sg] * 100, lw=1.1)
        for evm in TAXONOMY[sg][2]:
            if evm in m_ix:
                ax.axvline(x[m_ix[evm]], color="tab:orange", lw=1.2)
        ax.set_title(sg, fontsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("subgenre demand share of r/gamingsuggestions "
                 "(orange = subgenre-defining hit)")
    fig.tight_layout()
    fig.savefig(HERE / "exports" / "subgenre_demand_panel.png", dpi=120)
    S.to_csv(HERE / "exports" / "subgenre_monthly_share.csv")
    print("wrote subgenre_event_study.csv, subgenre_monthly_share.csv, "
          "subgenre_demand_panel.png")


if __name__ == "__main__":
    {"prefilter": prefilter, "classify": classify,
     "analyze": analyze}[sys.argv[1]]()
