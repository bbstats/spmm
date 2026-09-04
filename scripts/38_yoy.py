"""Out-of-season reliability: hold out one season, train on the seasons around it, predict it.

The decisive comparison between rating systems, and the first internal criterion in this project
that can see ATTRIBUTION -- whether credit is split correctly among five teammates.

Everything used before was blind to it.  Within-season stint MSE cannot see it, because the same
five players always appear together, so moving credit between them barely changes any lineup sum
(scripts/26_which_loss.py: prefers the wrong end at 3.4 standard errors while moving 0.065%).  The
next-window on-court benchmark cannot see it either, being another model built from the same margin
data with the same blind spot (FINDINGS.md section 13, where it ran to its grid boundary).

Predicting a season the ratings never saw breaks the degeneracy: rosters turn over, so a player
appears alongside different teammates and mis-split credit finally has somewhere to show up.  About
half of a returning player's teammate-possessions are with someone new, against effectively none for
a within-season split, so the test has real power against misattribution.  And what is predicted is
actual points scored -- ground truth, not another model's estimate.

    hold out season H
    train on a SYMMETRIC neighbourhood of H that excludes it: K=2 is {H-1, H+1},
        K=4 is {H-2, H-1, H+1, H+2}, and so on
    predict every stint of H from the ten players on the floor
    score = possession-weighted mean squared error in points per 100 possessions

Why symmetric-around-H rather than "train on a block, predict what flanks it":

  * The held-out season is IDENTICAL across every method and every K, so every comparison is exactly
    paired.  Training on a block and predicting its neighbours makes K=1 and K=3 score different
    seasons at different distances in time, which is not a comparison.
  * Aging cancels.  A player ages into H from below and out of it above, so a symmetric block
    brackets his ability in H instead of sitting to one side of it.
  * Varying K on fixed held-out data answers "how many seasons should a rating pool?" directly --
    more data against staler data -- which is the estimand-mismatch question HANDOFF left open.

Choices applied identically to every method, so none of them favours anyone:

  * The level is refit on the held-out season: an intercept and a home term by weighted least
    squares.  Scoring drifts across eras and that is not what is being tested.  It is the level, not
    the attribution, so it is not leakage.
  * Players absent from the training block score 0.
  * Reported at native scale AND after a fitted scalar, per side.  Raw error punishes a system whose
    ratings are simply too wide even when its ranking is perfect.  The scalar is fitted ON the
    held-out season, so those columns are an amplitude diagnostic and an upper bound, never a score.
  * Reported per stint AND aggregated to team-game, since stint error is mostly irreducible binomial
    noise and compresses the differences between methods.
  * Every prior is excluded from every window its training block or the held-out season touches.

Lambda is held FIXED across methods, because tuning it on this criterion and then reporting this
criterion would be circular.  Pass more than one to check the ranking survives the choice.

usage: python scripts/38_yoy.py [first=1998] [last=2025] [--k=2,4] [--lams=18352]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.windows import build_window, hybrid_beta, window_label, window_seasons  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
FE = cfg["features"]
nf = len(FE)


def _flag(name, default):
    hit = [a for a in sys.argv[1:] if a.startswith(f"--{name}=")]
    return [float(x) for x in hit[0].split("=")[1].split(",")] if hit else default


args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else 1998
LAST = int(args[1]) if len(args) > 1 else 2025
KS = [int(k) for k in _flag("k", [2, 4])]
LAMS = _flag("lams", [float(cfg["lam_plugin"])])
RATIO = float(cfg["lam_ratio_plugin"])
S0, S1 = int(cfg["first_season"]), int(cfg["last_season"])

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
# which 3-season window a season belongs to, so a prior can be kept off every window its training
# block or the held-out season touches
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}


def team_beta(labs):
    """The team-priced beta (stint margin on LINEUP SUMS), leave-those-windows-out.

    The published per-window beta was fit on that window, so using it where the window overlaps the
    training block or the held-out season would let this method see the answer while the hybrid's
    prior is already excluded from its own.  The coefficients are near-flat across eras, so averaging
    the remaining windows costs almost nothing.
    """
    d = base[~base.window.isin(labs)]
    o = d[d.side == "O"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    return np.concatenate([o, -dd])          # defense is stored flipped; the model wants it raw


# name -> (beta given the excluded window labels, target)
METHODS = {
    "1 RAPM, no prior": lambda labs: (np.zeros(2 * nf), "pts"),
    "2 PI-RAPM, team-priced": lambda labs: (team_beta(labs), "pts"),
    "3 hybrid": lambda labs: (hybrid_beta(panel, FE, labs), "pts"),
    "4 hybrid + xPTS(ft)": lambda labs: (hybrid_beta(panel, FE, labs), "xpts_ft"),
}

_cache: dict = {}


def design(seasons, target):
    key = (tuple(seasons), target)
    if key not in _cache:
        if len(_cache) > 4:
            _cache.clear()
        _cache[key] = build_window(list(seasons), cfg, target=target)
    return _cache[key]


def neighbourhood(h, k):
    """The K seasons nearest H, excluding H, balanced either side where the era allows."""
    out, d = [], 1
    while len(out) < k and d <= (S1 - S0):
        for s in (h - d, h + d):
            if S0 <= s <= S1 and len(out) < k:
                out.append(s)
        d += 1
    return sorted(out)


def ratings(seasons, beta, target, lam):
    """Per-player offense and defense from the training block, in the model's RAW sign."""
    wd = design(seasons, target)
    pipe = plugin_fit(wd, beta, lam=lam, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                         "o": ro @ beta[:nf] + mm.u_[:m],
                         "d": rd @ beta[nf:] + mm.u_[m:]})


def score(rat, h):
    """Predict the held-out season's stints from the ratings alone."""
    wd = design([h], "pts")             # ALWAYS scored against actual points
    m = wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    r = pd.DataFrame({"player_id": ids}).merge(rat, on="player_id", how="left").fillna(0.0)
    c_off = wd.X[:, :m] @ r["o"].to_numpy()
    c_def = wd.X[:, m:2 * m] @ r["d"].to_numpy()
    contrib = c_off + c_def
    home = np.asarray(wd.X[:, wd.spec.f_col("home")].todense()).ravel()
    y, w = wd.y, wd.w
    one = np.ones(len(y))

    def wls(cols, target):
        A = np.column_stack(cols)
        return A, np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ target)

    def wmse(pred):
        return float(np.average((y - pred) ** 2, weights=w))

    A, c = wls([one, home], y - contrib)
    pred = A @ c + contrib
    A0, c0 = wls([one, home], y)
    base_pred = A0 @ c0
    Ac, cc = wls([one, home, contrib], y)
    A2, c2 = wls([one, home, c_off, c_def], y)
    covered = float((r.o.to_numpy() != 0.0).mean())

    # team-game: sum predicted and actual points over each team's rows in a game, which cancels most
    # of the per-possession binomial noise
    g = pd.DataFrame({"g": wd.rows["game_idx"].to_numpy(), "h": wd.rows["is_home_off"].to_numpy(),
                      "poss": w, "act": y * w / 100.0, "prd": pred * w / 100.0,
                      "b": base_pred * w / 100.0}).groupby(["g", "h"]).sum()
    return dict(n=w.sum(), mse=wmse(pred), base=wmse(base_pred), calib=wmse(Ac @ cc),
                calib_side=wmse(A2 @ c2), scale=float(cc[2]),
                scale_off=float(c2[2]), scale_def=float(c2[3]), covered=covered,
                tg=float(np.average(((g.act - g.prd) / g.poss * 100) ** 2, weights=g.poss)),
                tg_base=float(np.average(((g.act - g.b) / g.poss * 100) ** 2, weights=g.poss)),
                tg_n=float(g.poss.sum()))


rows = []
t0 = time.time()
held = [h for h in range(FIRST, LAST + 1) if S0 <= h <= S1]
print(f"held-out seasons {held[0]}-{held[-1]} ({len(held)}), K {KS}, lambdas {LAMS}\n")
for h in held:
    for k in KS:
        tr = neighbourhood(h, k)
        labs = {WIN_OF[s] for s in tr + [h]}
        for name, spec in METHODS.items():
            beta, target = spec(labs)
            for lam in LAMS:
                r = score(ratings(tr, beta, target, lam), h)
                rows.append(dict(held_out=h, k=k, train=",".join(map(str, tr)),
                                 method=name, lam=lam, **r))
    print(f"  {h} done ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "yoy.parquet", index=False)


def summarise(d):
    n, tn = d.n.sum(), d.tg_n.sum()
    return pd.Series({
        "mse": float((d.mse * d.n).sum() / n),
        "vs_no_ratings": 1 - float((d.mse * d.n).sum() / (d.base * d.n).sum()),
        "calibrated": float((d.calib_side * d.n).sum() / n),
        "scale_off": float((d.scale_off * d.n).sum() / n),
        "scale_def": float((d.scale_def * d.n).sum() / n),
        "team_game": float((d.tg * d.tg_n).sum() / tn),
        "tg_vs_no_ratings": 1 - float((d.tg * d.tg_n).sum() / (d.tg_base * d.tg_n).sum()),
        "covered": float(d.covered.mean()),
    })


print("\n=== out-of-season reliability, pooled over every held-out season")
print("    mse            weighted MSE in points per 100 possessions -- LOWER IS BETTER")
print("    vs_no_ratings  share of held-out error the ratings remove -- HIGHER IS BETTER")
print("    scale_*        what the held-out season wants the ratings multiplied by; 1.0 = calibrated,")
print("                   below 1 = too wide.  Fitted out of sample: a diagnostic, not a score.")
for lam in LAMS:
    for k in KS:
        d = D[(D.lam == lam) & (D.k == k)]
        if not len(d):
            continue
        print(f"\n-- K = {k} training seasons, lambda {lam:.0f}")
        print(d.groupby("method", sort=True).apply(summarise, include_groups=False).to_string())

print("\n=== paired by held-out season, against the no-prior baseline (negative = better)")
base_col = sorted(METHODS)[0]
for lam in LAMS:
    for k in KS:
        p = D[(D.lam == lam) & (D.k == k)].pivot_table(index="held_out", columns="method", values="mse")
        if base_col not in p.columns or len(p) < 2:
            continue
        d = p.sub(p[base_col], axis=0).drop(columns=base_col)
        n = len(d)
        print(f"\n-- K = {k}, lambda {lam:.0f}  ({n} held-out seasons)")
        print(pd.DataFrame({"mean_diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(n),
                            "z": d.mean() / (d.std(ddof=1) / np.sqrt(n)),
                            "wins": (d < 0).sum().astype(int)}).to_string())

if len(KS) > 1:
    print("\n=== how many seasons should a rating pool? (same held-out data, directly comparable)")
    print(D[D.lam == LAMS[0]].pivot_table(index="method", columns="k", values="mse").to_string())

D.round(5).to_csv(OUT / "csv" / "yoy.csv", index=False)
print(f"\nwrote outputs/yoy.parquet and outputs/csv/yoy.csv  ({time.time() - t0:.0f}s)")
