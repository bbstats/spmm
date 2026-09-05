"""Player rating tables per season group: the box prior, the correction, the residual, the rating.

This is the canonical ratings table: one code path, one file.  scripts/07_plots.py reads it, and
outputs/ratings_boosted.parquet is an intermediate holding only the correction.

The shipped prior is the HYBRID (config `ratings_prior`): the box score priced to predict a PLAYER
on offense, and NO box prior at all on defense.  The box term is an offset `Xbox @ beta` with
separate offensive and defensive columns, so zeroing beta's defensive half is exactly "no defensive
prior" -- one fit, one penalty, no per-side machinery.  Against the consensus that is total 0.896 /
offense 0.879 / defense 0.888, against 0.784 / 0.851 / 0.755 for the team-priced prior it replaces.

One row per player per three-season window:
  prior_*    the player's padded three-year rates times the hybrid beta (zero on defense)
  boost_*    the LRBoost correction from scripts/10_boost.py, carried for comparison only: it is
             worth 0 on total and -0.020 on offense, so it is NOT in the shipped rating
  u_*        the RAPM residual (the whole defensive rating, since defense has no prior)
  u_plain_*  the RAPM residual fit with no correction at all
  rapm_mm_*  prior + u_plain, the rating the model would give without the boosted stage
  rating_*   prior + u, the shipped rating
  *_po       the same with the playoff coefficients (beta + delta) in place of beta
Offense, defense (flipped so positive = good) and total = offense + defense.
usage: python scripts/08_ratings.py [min_poss=1000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.boxtable import player_names, season_box  # noqa: E402
from eracoef.config import load_config  # noqa: E402
from eracoef.stints import season_names  # noqa: E402
from eracoef.windows import (build_window, hybrid_beta, player_ratings_table,  # noqa: E402
                             window_label, window_seasons)

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 2)
cfg = load_config()
MIN_POSS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = Path(cfg["_root"]) / "outputs"
CSV = OUT / "csv"
CSV.mkdir(parents=True, exist_ok=True)
FEATS = cfg["features"]
nf = len(FEATS)

coefs = pd.read_parquet(OUT / "coefs.parquet")
base = coefs[coefs.run == "base"]
po_path = OUT / "coefs_playoffs.parquet"
delta_pool = None
if po_path.exists():
    po = pd.read_parquet(po_path)
    # the playoff block pooled over the whole era: per-window delta has SEs of 0.18-0.87 per
    # coefficient, which a player's rates amplify into meaningless swings, and delta is flat
    # across eras anyway, so the pooled estimate is what the playoff ratings use
    delta_pool = po[(po["pool"] == "era") & (po["lam_delta"] == po["lam_delta"].min())]

PRIOR = cfg.get("ratings_prior", {})
USE_HYBRID = PRIOR.get("offense") == "player" and PRIOR.get("defense") == "none"
USE_BOOST = bool(PRIOR.get("boost", True))
# The role-prior chain (FINDINGS 19): no linear box prior at all; the per-player offset carries the
# Simple SPM (role and age) on both sides plus the boosted box prior on the sides set to "gbdt".
USE_CHAIN = PRIOR.get("role_prior") == "spm"
CHAIN_SIDES = tuple(s for s, key in (("O", "offense"), ("D", "defense")) if PRIOR.get(key) == "gbdt")
chain_ctx = None
if USE_CHAIN:
    from eracoef.holdout import Context
    from eracoef.spm import chain_offset
    chain_ctx = Context.load(cfg)
    if chain_ctx.rpanel is None or chain_ctx.role_inputs is None:
        raise SystemExit("ratings_prior.role_prior is 'spm' but outputs/role_panel.parquet or data/cache/roles.parquet "
                         "is missing; run scripts/49_role_panel.py")
    chain_fn = chain_offset(CHAIN_SIDES, mode=str(cfg.get("gbdt", {}).get("mode", "full")))
panel = None
if USE_HYBRID and not USE_CHAIN:
    xp = OUT / "xrapm_panel.parquet"
    if not xp.exists():
        raise SystemExit(f"{xp} is missing; run scripts/27_xrapm_prior.py first")
    panel = pd.read_parquet(xp)


def raw_beta(df, col="beta"):
    """Back out the model's raw 26-vector from the flipped, per-side table."""
    o = df[df.side == "O"].set_index("feature")[col].reindex(FEATS).to_numpy()
    d = df[df.side == "D"].set_index("feature")[col].reindex(FEATS).to_numpy()
    return np.concatenate([o, -d])          # D is stored flipped; the model wants it raw


# The nonlinear correction, if scripts/10_boost.py has run.  It has to go in as an OFFSET, not be
# added afterwards: the residual must be refit knowing the correction, or it double-counts whatever
# the correction already explains.  Adding it on top instead moved players by up to 1.4 points.
bp = OUT / "ratings_boosted.parquet"
boost = None
if bp.exists():
    boost = pd.read_parquet(bp)[["window", "player_id", "boost_off", "boost_def", "boost_total"]]


def offset_for(wd, lab):
    """The correction as one value per Z column, in the model's raw sign, aligned to this window."""
    if boost is None:
        return None
    ids = pd.DataFrame({"player_id": wd.spec.ps_table["player_id"].to_numpy()})
    b = ids.merge(boost[boost.window == lab], on="player_id", how="left").fillna({"boost_off": 0.0, "boost_def": 0.0})
    # defense is stored flipped so positive = good; the model wants it raw
    return np.concatenate([b.boost_off.to_numpy(), -b.boost_def.to_numpy()])


out = []
t0 = time.time()
for w in window_seasons(cfg):
    seasons = list(range(w[0], w[1] + 1))
    lab = window_label(seasons)
    # the target the ratings are fit to: "xpts_ft" replaces made free throws by the shooter's
    # expectation (FINDINGS.md sections 16-18: the one luck adjustment that beat actual points)
    wd = build_window(seasons, cfg, target=PRIOR.get("target", "pts"))
    chain = None
    if USE_CHAIN:
        # the whole prior is the offset; the block's own window is the only label to keep it off
        beta, beta_po = np.zeros(2 * nf), None
        chain = np.asarray(chain_fn(seasons, chain_ctx, wd), dtype=float)
        m_ = wd.spec.n_ps
        chain_parts = (chain[:m_], chain[m_:])
    elif USE_HYBRID:
        beta = hybrid_beta(panel, FEATS, lab)
        # the playoff delta was fit as a change in the TEAM-priced coefficients; apply it to the
        # offensive half only, or the defensive half stops being zero and the playoff variant
        # quietly reintroduces the defensive prior the hybrid exists to remove
        beta_po = None
        if delta_pool is not None:
            d = raw_beta(delta_pool, "delta")
            beta_po = np.concatenate([beta[:nf] + d[:nf], np.zeros(nf)])
    else:
        beta = raw_beta(base[base.window == lab])
        beta_po = beta + raw_beta(delta_pool, "delta") if delta_pool is not None else None
    names = player_names(season_box(seasons, ["RS"], cfg),
                         pd.concat([season_names(s, "RS", cfg) for s in seasons], ignore_index=True))
    r = player_ratings_table(wd, beta, cfg, seasons, beta_po=beta_po, names=names,
                             prior_offset=chain if USE_CHAIN else (offset_for(wd, lab) if USE_BOOST else None),
                             prior_parts=chain_parts if USE_CHAIN else None)
    # A second fit for the DEFENSIVE side on its own target (config ratings_prior.defense_target):
    # "x3def" replaces every opponent three-point make by 3 x the shooter's padded 3P%, so the
    # defenders' coefficients are not charged for whether an open three dropped (FINDINGS.md 18).
    # Offense keeps the fit above.  Every *_def column comes from this fit; totals are re-summed.
    DEF_TARGET = PRIOR.get("defense_target")
    if DEF_TARGET:
        from eracoef.xshoot import DEFENSE_TARGETS
        wd_pts = wd if PRIOR.get("target", "pts") == "pts" else build_window(seasons, cfg)
        wd_d, rep_d = DEFENSE_TARGETS[DEF_TARGET](seasons, cfg, wd_pts)
        r_d = player_ratings_table(wd_d, beta, cfg, seasons, beta_po=beta_po, names=None,
                                   prior_offset=chain if USE_CHAIN else (offset_for(wd_d, lab) if USE_BOOST else None),
                                   prior_parts=chain_parts if USE_CHAIN else None)
        dcols = [c for c in r_d.columns if "_def" in c]
        r = r.drop(columns=dcols).merge(r_d[["player_id", *dcols]], on="player_id", how="left")
        for part in ("prior", "u", "u_plain", "rapm_mm"):
            r[f"{part}_total"] = r[f"{part}_off"] + r[f"{part}_def"]
        if "prior_total_po" in r.columns:
            r["prior_total_po"] = r.prior_off_po + r.prior_def_po
            r["rapm_mm_total_po"] = r.rapm_mm_off_po + r.rapm_mm_def_po
        print(f"    defense from target {DEF_TARGET}: k3 = {rep_d.get('k3', float('nan')):.0f}, "
              f"3PM ratio {rep_d['gates'].r3.mean():.4f}", flush=True)
        del wd_d
    out.append(r)
    print(f"  {lab}: {len(r)} players ({time.time() - t0:.0f}s)", flush=True)
    del wd

rat = pd.concat(out, ignore_index=True)

if boost is not None:
    rat = rat.merge(boost, on=["window", "player_id"], how="left")
    for c in ("boost_off", "boost_def", "boost_total"):
        rat[c] = rat[c].fillna(0.0)

# The rating.  `u` is already the residual refit given whatever offset went in, so the correction is
# added here only when it was also carried into the fit -- adding it on top of a residual fit without
# it double-counts what it explains and moved players by up to 1.4 points.
g = (lambda side: rat[f"boost_{side}"]) if (boost is not None and USE_BOOST) else (lambda side: 0.0)
for side in ("off", "def", "total"):
    rat[f"rating_{side}"] = rat[f"prior_{side}"] + g(side) + rat[f"u_{side}"]
    if f"prior_{side}_po" in rat.columns:
        rat[f"rating_{side}_po"] = rat[f"prior_{side}_po"] + g(side) + rat[f"u_{side}"]

# The rank-calibration map (FINDINGS.md section 18).  Out of season, the held-out seasons want the
# worst offensive decile multiplied by 0.77 and the best by 1.01, and the same for defense read in
# the good direction: good players are calibrated, bad ones exaggerated.  The map is a monotone
# per-side curve fitted on the pooled decile slopes of every held-out season (holdout.fit_rank_map),
# scored leave-one-season-out at -0.4 points per 100 at team-game level (z -6, 26 of 28 seasons).
# It preserves every within-side rank; the consensus read is unchanged (0.894 against 0.895).
RM = PRIOR.get("rank_map")
if RM:
    from eracoef.holdout import fit_rank_map
    rp_rank = Path(cfg["_root"]) / RM["table"]
    if not rp_rank.exists():
        raise SystemExit(f"{rp_rank} is missing; run scripts/45_holdout.py --systems={RM['system']} --rank --tag=final")
    rank_table = pd.read_parquet(rp_rank)
    f_o = fit_rank_map(rank_table, RM["system"], "o", int(RM["k"]))
    f_d = fit_rank_map(rank_table, RM["system"], "d", int(RM["k"]))
    for suffix in ("", "_po"):
        if f"rating_off{suffix}" not in rat.columns:
            continue
        rat[f"rating_off{suffix}_raw"] = rat[f"rating_off{suffix}"]
        rat[f"rating_def{suffix}_raw"] = rat[f"rating_def{suffix}"]
        rat[f"rating_off{suffix}"] = f_o(rat[f"rating_off{suffix}"].to_numpy())
        rat[f"rating_def{suffix}"] = -f_d(-rat[f"rating_def{suffix}"].to_numpy())   # the map is in raw sign
        rat[f"rating_total{suffix}"] = rat[f"rating_off{suffix}"] + rat[f"rating_def{suffix}"]
    print(f"rank map applied from {RM['table']} ({RM['system']}, K={RM['k']})")

SORT = "rating_total" if "rating_total" in rat.columns else "rapm_mm_total"
cols = ["window", "window_mid", "season", "player_id", "player_name", "poss_off", "poss_def", "shrinkage",
        "prior_off", "prior_def", "prior_total", "boost_off", "boost_def", "boost_total",
        "u_off", "u_def", "u_total", "u_plain_off", "u_plain_def", "u_plain_total",
        "rapm_mm_off", "rapm_mm_def", "rapm_mm_total",
        "rating_off", "rating_def", "rating_total"]
cols += [c for c in rat.columns if (c.endswith("_po") or c.endswith("_raw")) and c not in cols]
rat = rat[[c for c in cols if c in rat.columns]].sort_values(["window", SORT], ascending=[True, False])
rat.to_parquet(OUT / "player_ratings.parquet", index=False)
rat.round(4).to_csv(CSV / "player_ratings.csv", index=False)

# player_ratings_by_window is gone: with player_unit "window" the rollup was the identity, so the
# two files held the same table twice.
for stale in (OUT / "player_ratings_by_window.parquet", CSV / "player_ratings_by_window.csv"):
    if stale.exists():
        stale.unlink()
        print("removed", stale.name)

print("")
print(f"wrote {len(rat)} player-window rows")
SHOW = [c for c in ("player_name", "poss_off", "prior_total", "boost_total", "u_total",
                    "rating_off", "rating_def", SORT) if c in rat.columns]
for lab in rat.window.unique():
    g = rat[(rat.window == lab) & (rat.poss_off >= MIN_POSS)]
    print("")
    print(f"=== {lab}: top 10 (>= {MIN_POSS} possessions, n = {len(g)}) ===")
    print(g.head(10)[SHOW].to_string(index=False))
print("")
print("correlation of the box prior with the ridge residual (u), by window:")
print(rat.groupby("window").apply(lambda g: pd.Series({
    "corr_prior_u_total": np.corrcoef(g.prior_total, g.u_total)[0, 1],
    "sd_prior": g.prior_total.std(),
    "sd_boost": g.boost_total.std() if "boost_total" in g.columns else 0.0,
    "sd_u": g.u_total.std(), "sd_rating": g[SORT].std()}),
    include_groups=False).round(3).to_string())
