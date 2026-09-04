"""Year-over-year reliability: fit one season, predict the seasons before and after.

The decisive comparison between rating systems, and the first internal criterion in this project
that can see ATTRIBUTION -- whether credit is split correctly among five teammates.

Everything used before was blind to it.  Within-season stint MSE cannot see it, because the same
five players always appear together, so moving credit between them barely changes any lineup sum
(scripts/26_which_loss.py: prefers the wrong end at 3.4 standard errors while moving 0.065%).  The
next-window on-court benchmark cannot see it either, because it is another model built from the same
margin data with the same blind spot (FINDINGS.md section 13, where it ran to its grid boundary).

Predicting a DIFFERENT season breaks the degeneracy: the player turns up alongside different
teammates, so mis-split credit finally has somewhere to show up.  And the thing being predicted is
actual points scored, which is ground truth rather than another model's estimate.

    fit ratings on season Y (offense and defense only)
    predict every stint of Y-1 and Y+1 from the ten players on the floor
    score = possession-weighted mean squared error in points per 100 possessions

Predicting BOTH directions is deliberate: a player ages into Y from Y-1 and out of Y into Y+1, so
the two biases point opposite ways and largely cancel.  Bookend seasons are skipped.

Four choices, applied identically to every method so none of them favours anyone:

  * The level is refit on the held-out season.  Only an intercept and a home term, by weighted least
    squares -- scoring drifts across eras and that is not what is being tested.  It is the level, not
    the attribution, so it is not leakage.
  * Players absent from the training season score 0.
  * Reported at native scale AND after one global scalar.  Raw error punishes a system whose ratings
    are simply too wide even when its ranking is perfect, and the pre-hybrid board's defensive spread
    was 2.13x too wide.  The gap between the two separates mis-calibrated from mis-ranked.  The
    scalar is fitted ON THE HELD-OUT SEASON, so the calibrated column is an upper bound and a
    diagnostic, never the score.
  * Reported per stint AND aggregated to team-game.  Stint error is mostly irreducible binomial
    noise, which compresses the differences between methods.

Lambda is held FIXED across methods rather than tuned per method, because tuning it on this
criterion and then reporting this criterion would be circular.  The sweep over a small lambda grid
is there to show whether the ranking survives the choice.

usage: python scripts/38_yoy.py [first=1998] [last=2025] [--lams=6000,18352,50000]
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
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else 1998
LAST = int(args[1]) if len(args) > 1 else 2025
lam_arg = [a for a in sys.argv[1:] if a.startswith("--lams=")]
LAMS = ([float(x) for x in lam_arg[0].split("=")[1].split(",")] if lam_arg
        else [6000.0, float(cfg["lam_plugin"]), 50000.0])
RATIO = float(cfg["lam_ratio_plugin"])

panel = pd.read_parquet(OUT / "xrapm_panel.parquet")
coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]


def team_beta(lab):
    """The team-priced beta (stint margin on LINEUP SUMS), LEAVE-ONE-WINDOW-OUT.

    The published per-window beta was fit on that window, which contains the training season AND
    both held-out seasons -- using it directly would let this method see the answer while the
    hybrid's beta (windows.hybrid_beta) is already excluded from its own window.  Averaging the
    other windows' betas keeps the two comparable.  The coefficients are near-flat across eras, so
    this costs almost nothing in accuracy.
    """
    d = base[base.window != lab]
    o = d[d.side == "O"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    dd = d[d.side == "D"].groupby("feature")["beta"].mean().reindex(FE).to_numpy()
    return np.concatenate([o, -dd])          # defense is stored flipped; the model wants it raw


# which 3-season window a season belongs to, so the leave-one-window-out player-priced beta can be
# reused for a single-season fit without ever being fit on that season
WIN_OF = {s: window_label(list(range(w[0], w[1] + 1)))
          for w in window_seasons(cfg) for s in range(w[0], w[1] + 1)}

# name -> (beta for this season, target).  beta is the raw 26-vector the model wants.
METHODS = {
    "1 RAPM, no prior": lambda s: (np.zeros(2 * nf), "pts"),
    "2 PI-RAPM, team-priced": lambda s: (team_beta(WIN_OF[s]), "pts"),
    "3 hybrid": lambda s: (hybrid_beta(panel, FE, WIN_OF[s]), "pts"),
    "4 hybrid + xPTS(ft)": lambda s: (hybrid_beta(panel, FE, WIN_OF[s]), "xpts_ft"),
}

_cache: dict = {}


def design(season, target):
    key = (season, target)
    if key not in _cache:
        if len(_cache) > 6:
            _cache.clear()
        _cache[key] = build_window([season], cfg, target=target)
    return _cache[key]


def ratings(season, beta, target, lam):
    """Per-player offense and defense from one season, in the model's RAW sign."""
    wd = design(season, target)
    pipe = plugin_fit(wd, beta, lam=lam, lam_ratio=RATIO, pad_target=cfg["pad_target"])
    exp, mm = pipe["exposure"], pipe["mm"]
    m = wd.spec.n_ps
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    return pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy(),
                         "o": ro @ beta[:nf] + mm.u_[:m],
                         "d": rd @ beta[nf:] + mm.u_[m:]})


def score(rat, season):
    """Predict a held-out season's stints from the ratings alone.  Returns a dict of errors."""
    wd = design(season, "pts")          # ALWAYS scored against actual points
    m = wd.spec.n_ps
    ids = wd.spec.ps_table["player_id"].to_numpy()
    r = pd.DataFrame({"player_id": ids}).merge(rat, on="player_id", how="left").fillna(0.0)
    c_off = wd.X[:, :m] @ r["o"].to_numpy()
    c_def = wd.X[:, m:2 * m] @ r["d"].to_numpy()
    contrib = c_off + c_def
    home = np.asarray(wd.X[:, wd.spec.f_col("home")].todense()).ravel()
    y, w = wd.y, wd.w

    def fit_level(c):
        """Intercept and home term by weighted least squares on the held-out season."""
        A = np.column_stack([np.ones(len(y)), home])
        resid = y - c
        coef = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ resid)
        return A @ coef

    def wmse(pred):
        return float(np.average((y - pred) ** 2, weights=w))

    base = wmse(fit_level(np.zeros_like(y)))
    native = wmse(fit_level(contrib) + contrib)
    # the best single scalar on the ratings, fitted on the HELD-OUT season: an upper bound and an
    # amplitude diagnostic, never the score
    A = np.column_stack([np.ones(len(y)), home, contrib])
    coef = np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ y)
    calib = wmse(A @ coef)
    # separately per side: a single scalar hides offense and defense being wrong by different amounts
    A2 = np.column_stack([np.ones(len(y)), home, c_off, c_def])
    c2 = np.linalg.solve((A2 * w[:, None]).T @ A2, (A2 * w[:, None]).T @ y)

    # team-game: sum predicted and actual points over each team's rows in a game, which cancels most
    # of the per-possession binomial noise
    pred = fit_level(contrib) + contrib
    g = pd.DataFrame({"g": wd.rows["game_idx"].to_numpy(), "h": wd.rows["is_home_off"].to_numpy(),
                      "poss": w, "act": y * w / 100.0, "prd": pred * w / 100.0,
                      "b": (fit_level(np.zeros_like(y))) * w / 100.0}).groupby(["g", "h"]).sum()
    tg = float(np.average(((g.act - g.prd) / g.poss * 100) ** 2, weights=g.poss))
    tg_base = float(np.average(((g.act - g.b) / g.poss * 100) ** 2, weights=g.poss))
    return dict(sse=native * w.sum(), n=w.sum(), mse=native, base=base, calib=calib,
                scale=float(coef[2]), scale_off=float(c2[2]), scale_def=float(c2[3]),
                calib_side=wmse(A2 @ c2), tg=tg, tg_base=tg_base, tg_n=float(g.poss.sum()))


rows = []
t0 = time.time()
seasons = [s for s in range(FIRST, LAST + 1) if s - 1 >= cfg["first_season"] and s + 1 <= cfg["last_season"]]
print(f"training seasons {seasons[0]}-{seasons[-1]} ({len(seasons)}), lambdas {LAMS}\n")
for s in seasons:
    for name, spec in METHODS.items():
        beta, target = spec(s)
        for lam in LAMS:
            rat = ratings(s, beta, target, lam)
            for h in (s - 1, s + 1):
                r = score(rat, h)
                rows.append(dict(train=s, held_out=h, method=name, lam=lam, **r))
    print(f"  {s} done ({time.time() - t0:.0f}s)", flush=True)

D = pd.DataFrame(rows)
D.to_parquet(OUT / "yoy.parquet", index=False)


def summarise(d):
    """Possession-weighted pooling across all held-out season-pairs."""
    n = d.n.sum()
    return pd.Series({
        "mse": float((d.mse * d.n).sum() / n),
        "vs_no_ratings": 1 - float((d.mse * d.n).sum() / (d.base * d.n).sum()),
        "calibrated_mse": float((d.calib * d.n).sum() / n),
        "best_scale": float((d.scale * d.n).sum() / n),
        "scale_off": float((d.scale_off * d.n).sum() / n),
        "scale_def": float((d.scale_def * d.n).sum() / n),
        "team_game_mse": float((d.tg * d.tg_n).sum() / d.tg_n.sum()),
        "tg_vs_no_ratings": 1 - float((d.tg * d.tg_n).sum() / (d.tg_base * d.tg_n).sum()),
    })


print("\n=== year-over-year reliability, pooled over every held-out season")
print("    mse            weighted MSE in points per 100 possessions -- LOWER IS BETTER")
print("    vs_no_ratings  share of held-out error the ratings remove -- HIGHER IS BETTER")
print("    best_scale     the scalar the held-out season wants; 1.0 = correctly calibrated,")
print("                   below 1 = ratings too wide.  Diagnostic only, fitted out of sample.")
for lam in LAMS:
    print(f"\n-- lambda {lam:.0f}")
    t = D[D.lam == lam].groupby("method", sort=False).apply(summarise, include_groups=False)
    print(t.to_string())

print("\n=== paired by held-out season, against the no-prior baseline (negative = better)")
piv = D.pivot_table(index=["lam", "train", "held_out"], columns="method", values="mse")
base_col = list(METHODS)[0]
for lam in LAMS:
    p = piv.loc[lam]
    d = (p.sub(p[base_col], axis=0)).drop(columns=base_col)
    n = len(d)
    print(f"\n-- lambda {lam:.0f}  ({n} held-out season-pairs)")
    print(pd.DataFrame({"mean_diff": d.mean(), "se": d.std(ddof=1) / np.sqrt(n),
                        "z": d.mean() / (d.std(ddof=1) / np.sqrt(n)),
                        "wins": (d < 0).sum().astype(int)}).to_string())

D.round(5).to_csv(OUT / "csv" / "yoy.csv", index=False)
print(f"\nwrote outputs/yoy.parquet and outputs/csv/yoy.csv  ({time.time() - t0:.0f}s)")
