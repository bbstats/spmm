"""Build the role panel: APM -> Simple SPM -> RAPM_1, per player-window-side (src/eracoef/spm.py).

Per window, per side (offense on the free-throw target, defense on the opponent-3PM-replaced one,
like the board):
  1. APM        the plugin ridge with beta = 0 at the tiny penalty (config spm.apm_lam)
  2. Simple SPM the possession-weighted ridge of APM on the seven role-and-age inputs, fit on every
                OTHER window (leave-window-out), predicted for this one and possession-centred
  3. RAPM_1     the shipped ridge with prior_offset = Simple SPM; u and the shrinkage a per player

Writes outputs/role_panel.parquet with, per row: window, side, player_id, ps_idx, season (the year
the player played most in the window), poss, the 13 centred padded rates, share, gs_pct, age, apm,
spm, u, a, rapm1 (= spm + u).  Raw sign on both sides throughout.  outputs/xrapm_panel.parquet, the
reference systems' input, is asserted unchanged.

--check   also fit APM at config spm.apm_check_lams and print the SPM coefficients side by side
          (outputs/csv/spm_lambda_check.csv); the coefficients must not depend on the APM penalty.
usage: python scripts/49_role_panel.py [--check]
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.cv import plugin_fit  # noqa: E402
from eracoef.design import FEATURES  # noqa: E402
from eracoef.roles import RAW_INPUTS, build_roles, player_season_inputs, window_inputs  # noqa: E402
from eracoef.spm import (apm_fit, apm_lambda_check, fit_spm, panel_inputs_report, season_of_units,  # noqa: E402
                         spm_predict)
from eracoef.windows import build_window, window_label, window_seasons  # noqa: E402
from eracoef.xshoot import DEFENSE_TARGETS  # noqa: E402

pd.set_option("display.width", 250, "display.max_columns", 60, "display.precision", 3)
cfg = load_config()
OUT = Path(cfg["_root"]) / "outputs"
(OUT / "csv").mkdir(exist_ok=True)
S = cfg.get("spm", {})
LAM_P, RAT_P = float(cfg["lam_plugin"]), float(cfg["lam_ratio_plugin"])
CAP = float(cfg.get("roles", {}).get("share_cap", 0.9))
CHECK = "--check" in sys.argv
CHECK_LAMS = [float(x) for x in S.get("apm_check_lams", [30, 100, 300])] if CHECK else []
PANEL_PATH = Path(cfg["_root"]) / cfg.get("paths", {}).get("role_panel", "outputs/role_panel.parquet")
XP = OUT / "xrapm_panel.parquet"
h_before = hashlib.sha256(XP.read_bytes()).hexdigest() if XP.exists() else None

roles = build_roles(cfg)
inputs = player_season_inputs(roles, cap=CAP)
print(f"roles: {len(roles)} player-season-team rows, {inputs.player_id.nunique()} players, "
      f"age imputed {inputs.age_imputed.mean():.1%} of player-seasons", flush=True)


def designs(seasons):
    wd_pts = build_window(seasons, cfg)
    wd_o = build_window(seasons, cfg, target="xpts_ft")
    wd_d, rep = DEFENSE_TARGETS["x3def"](seasons, cfg, wd_pts)
    del wd_pts
    return wd_o, wd_d


# ------------------------------------------------------------------ pass 1: APM and the inputs
t0 = time.time()
parts = []
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd_o, wd_d = designs(seasons)
    inp = window_inputs(wd_o, inputs, cap=CAP)
    season = season_of_units(wd_o)
    for side, wd in (("O", wd_o), ("D", wd_d)):
        a = apm_fit(wd, cfg)
        R = a["ro"] if side == "O" else a["rd"]
        d = pd.DataFrame(R, columns=FEATURES)
        d.insert(0, "window", lab)
        d.insert(1, "side", side)
        d.insert(2, "player_id", wd.spec.ps_table["player_id"].to_numpy())
        d.insert(3, "ps_idx", np.arange(wd.spec.n_ps))
        d.insert(4, "season", season)
        d["poss"] = a["poss_o"] if side == "O" else a["poss_d"]
        for c in RAW_INPUTS:
            d[c] = inp[c].to_numpy()
        d["apm"] = a["u_o"] if side == "O" else a["u_d"]
        for lam in CHECK_LAMS:
            if lam == a["lam"]:
                d[f"apm_{lam:g}"] = d["apm"]
            else:
                b = apm_fit(wd, cfg, lam=lam)
                d[f"apm_{lam:g}"] = b["u_o"] if side == "O" else b["u_d"]
        parts.append(d)
    print(f"  pass 1 {lab}: {wd_o.spec.n_ps} players, inputs missing for {inp.attrs['n_missing']} of "
          f"{inp.attrs['n_psx']} player-seasons ({time.time() - t0:.0f}s)", flush=True)
    del wd_o, wd_d
P = pd.concat(parts, ignore_index=True)

# ------------------------------------------------------------------ pass 2: Simple SPM, leave-window-out
coef_rows = []
P["spm"] = 0.0
for lab in sorted(P.window.unique()):
    for side in ("O", "D"):
        fit = fit_spm(P, side, exclude={lab}, pen=float(S.get("pen", 1.0)), min_poss=float(S.get("min_poss", 500)))
        sel = (P.window == lab) & (P.side == side)
        g = spm_predict(fit, P.loc[sel, "share"], P.loc[sel, "gs_pct"], P.loc[sel, "age"])
        wgt = np.maximum(P.loc[sel, "poss"].to_numpy(dtype=float), 0.0)
        if wgt.sum() > 0:
            g = g - np.average(g, weights=wgt)
        P.loc[sel, "spm"] = g
        coef_rows.append(fit.table().assign(window=lab, n=fit.n))
coefs = pd.concat(coef_rows, ignore_index=True)
coefs.to_csv(OUT / "csv" / "spm_coefs.csv", index=False)
print("\n=== Simple SPM coefficients per unit of the raw input, mean over the leave-one-out fits (raw sign)")
print(coefs.pivot_table(index="input", columns="side", values="coef_raw", aggfunc=["mean", "std"]).round(4).to_string())

if CHECK:
    print("\n=== APM penalty check: SPM coefficients at each penalty (must agree)")
    tabs = []
    for side in ("O", "D"):
        panels = {lam: P.assign(apm=P[f"apm_{lam:g}"]) for lam in CHECK_LAMS}
        t = apm_lambda_check(panels, side, pen=float(S.get("pen", 1.0)), min_poss=float(S.get("min_poss", 500)))
        print(f"\n-- {side}")
        print(t.round(4).to_string())
        tabs.append(t.assign(side=side))
    pd.concat(tabs).to_csv(OUT / "csv" / "spm_lambda_check.csv")

# ------------------------------------------------------------------ pass 3: RAPM_1 with the SPM as the offset
P["u"], P["a"] = 0.0, 0.0
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    wd_o, wd_d = designs(seasons)
    m = wd_o.spec.n_ps
    so = P[(P.window == lab) & (P.side == "O")].sort_values("ps_idx")["spm"].to_numpy()
    sd = P[(P.window == lab) & (P.side == "D")].sort_values("ps_idx")["spm"].to_numpy()
    assert len(so) == m and len(sd) == m
    offset = np.concatenate([so, sd])
    nf = len(FEATURES)
    for side, wd in (("O", wd_o), ("D", wd_d)):
        fit = plugin_fit(wd, np.zeros(2 * nf), lams=[LAM_P], cv=2, lam_ratio=RAT_P, pad_target=cfg["pad_target"],
                         prior_offset=offset)
        mm = fit["mm"]
        a = mm.shrinkage_diag()
        sel = (P.window == lab) & (P.side == side)
        order = P.loc[sel].sort_values("ps_idx").index
        P.loc[order, "u"] = mm.u_[:m] if side == "O" else mm.u_[m:]
        P.loc[order, "a"] = a[:m] if side == "O" else a[m:]
    print(f"  pass 3 {lab} ({time.time() - t0:.0f}s)", flush=True)
    del wd_o, wd_d
P["rapm1"] = P["spm"] + P["u"]

cols = ["window", "side", "player_id", "ps_idx", "season", "poss", *FEATURES, *RAW_INPUTS, "apm", "spm", "u", "a", "rapm1"]
P[cols].to_parquet(PANEL_PATH, index=False)
if h_before is not None:
    assert hashlib.sha256(XP.read_bytes()).hexdigest() == h_before, "xrapm_panel.parquet changed; the reference must not move"

rep = panel_inputs_report(P)
print("\n=== per window and side: possession-weighted sd of apm, spm, u and rapm1 (points per 100)")
print(rep[["window", "side", "n", "apm_sd", "spm_sd", "u_sd", "rapm1_sd", "share_mean", "gs_pct_mean", "age_mean"]]
      .round(3).to_string(index=False))
rep.to_csv(OUT / "csv" / "role_panel_report.csv", index=False)
print(f"\nwrote {PANEL_PATH.relative_to(cfg['_root'])} ({len(P)} rows) and outputs/csv/spm_coefs.csv  ({time.time() - t0:.0f}s)")
