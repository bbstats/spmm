"""Which loss can actually select the defensive prior weight?

`c_def` scales the defensive box prior in the plug-in offset.  scripts/25 shows the consensus wants
c_def = 0 and the shipped value is 1.0.  The question this script settles is *why the pipeline never
noticed*: held-out stint mean squared error, the loss every constant in this project was chosen on,
is evaluated on rows, and a row only ever sees the LINEUP SUM of player effects.  Moving credit
between two players who share the floor leaves that sum unchanged, so the loss is nearly flat along
exactly the direction `c_def` moves.

Two losses, same folds, same everything else:

  A. held-out weighted stint MSE, five game-grouped folds   (the loss we used)
  B. rank agreement with the external consensus              (a player-level loss)

If A is flat or prefers c_def = 1 while B prefers c_def = 0, then A cannot select this parameter,
and no amount of care in applying A would ever have caught the defect.

usage: python scripts/26_which_loss.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import crossfit_beta, plugin_fit  # noqa: E402
from eracoef.windows import build_window  # noqa: E402

pd.set_option("display.width", 250, "display.precision", 4)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
nf = len(cfg["features"])
LAM_B, RAT_B = float(cfg["lam_beta"]), float(cfg["lam_ratio_beta"])
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
C_DEF = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
SEASONS = [2024, 2025, 2026]

con = pd.read_parquet(OUT / "vs_consensus.parquet")[
    ["player_id", "adj_offense", "adj_defense", "adj_overall"]]
wd = build_window(SEASONS, cfg)
m_ps = wd.spec.n_ps
ids = wd.spec.ps_table["player_id"].to_numpy()

# ---- loss A: held-out stint MSE, five game-grouped folds
rows = []
t0 = time.time()
for k, (tr, te) in enumerate(GroupKFold(5).split(wd.X, wd.y, wd.groups)):
    tr_mask = np.zeros(len(wd.y), bool)
    tr_mask[tr] = True
    cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, mask=tr_mask,
                       pad_target=cfg["pad_target"])
    for c in C_DEF:
        b = cf.beta_.copy()
        b[nf:] = cf.beta_[nf:] * c
        pipe = plugin_fit(wd, b, lam=LAM_P, lam_ratio=RAT_P, mask=tr_mask,
                          pad_target=cfg["pad_target"])
        p = pipe.predict(wd.X[te])
        rows.append(dict(fold=k, c_def=c,
                         mse=float(np.average((wd.y[te] - p) ** 2, weights=wd.w[te]))))
    print(f"  fold {k} done ({time.time() - t0:.0f}s)", flush=True)
A = pd.DataFrame(rows)

# ---- loss B: rank agreement with the consensus, full-window fit
rows = []
cf = crossfit_beta(wd, lams=[LAM_B], cv=2, lam_ratio=RAT_B, pad_target=cfg["pad_target"])
for c in C_DEF:
    b = cf.beta_.copy()
    b[nf:] = cf.beta_[nf:] * c
    fit = plugin_fit(wd, b, lam=LAM_P, lam_ratio=RAT_P, pad_target=cfg["pad_target"])
    mm, exp = fit["mm"], fit["exposure"]
    ro = exp.season_rates_ - exp.means_o_ / 5.0
    rd = exp.season_rates_d_ - exp.means_d_ / 5.0
    g = pd.DataFrame({"player_id": ids,
                      "off": ro @ b[:nf] + mm.u_[:m_ps],
                      "dfn": -(rd @ b[nf:]) - mm.u_[m_ps:]}).merge(con, on="player_id")
    rows.append(dict(c_def=c, rank_total=spearmanr(g.off + g.dfn, g.adj_overall).statistic,
                     rank_def=spearmanr(g.dfn, g.adj_defense).statistic))
B = pd.DataFrame(rows)

# ---- fold-paired, because the fold-to-fold level dwarfs the difference being tested
base = A[A.c_def == 1.0].set_index("fold").mse
A["vs_shipped"] = A.mse - A.fold.map(base)          # negative = better than shipped
tab = (A.groupby("c_def").agg(mse=("mse", "mean"), d=("vs_shipped", "mean"),
                              se=("vs_shipped", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
       .reset_index().merge(B, on="c_def"))
tab["z"] = tab.d / tab.se.replace(0, np.nan)

print("\n=== the same parameter under two losses")
print("   `d` is held-out stint MSE minus the shipped setting, fold-paired.")
print("   Negative d = better stint prediction.  Higher rank = better player ranking.")
print(tab[["c_def", "mse", "d", "se", "z", "rank_total", "rank_def"]].round(4).to_string(index=False))

a_best = tab.loc[tab.mse.idxmin(), "c_def"]
b_best = tab.loc[tab.rank_total.idxmax(), "c_def"]
spread = tab.mse.max() - tab.mse.min()
print(f"\n   stint MSE picks   c_def = {a_best}   (total range across the whole sweep: {spread:.3f})")
print(f"   player rank picks c_def = {b_best}   (range: {tab.rank_total.max() - tab.rank_total.min():.3f})")
print(f"\n   Held-out stint MSE moves by {spread:.3f} across a parameter that moves rank agreement")
print(f"   by {tab.rank_total.max() - tab.rank_total.min():.3f}.")
print("   The loss the pipeline was tuned on is nearly blind to the parameter that broke it.")
tab.to_csv(OUT / "csv" / "which_loss.csv", index=False)
print("\nwrote outputs/csv/which_loss.csv")
