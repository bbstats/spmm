"""Step 6: all ten windows plus robustness runs -> outputs/coefs.parquet.
usage: python scripts/04_fit_all.py [runs=base,lam_x3,lam_div3,gt_half,gt_zero,margin_bins,rolling]
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config
from eracoef.windows import run_all, write_outputs

pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
cfg = load_config()
runs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["base", "lam_x3", "lam_div3", "gt_half", "gt_zero", "margin_bins", "rolling"]
print(f"runs: {runs}\nlam_beta {cfg['lam_beta']} ratio {cfg['lam_ratio_beta']} pad_target {cfg['pad_target']}")
res = run_all(cfg, runs=runs)
paths = write_outputs(res, cfg)
print("\nwrote:", {k: str(v) for k, v in paths.items()})
c = res["coefs"]
print(f"\ncoefs rows: {len(c)}  runs: {sorted(c.run.unique())}  windows: {c.window.nunique()}")
if len(res["variance"]):
    print("\nshare of impact explained by the box prior, per window:")
    print(res["variance"][["window", "share_O", "share_D", "tau2_O_zero", "tau2_D_zero", "tau2_O_plugin", "tau2_D_plugin"]].round(3).to_string(index=False))
b = c[c.run == "base"]
if len(b):
    print("\nbase beta, offense:")
    print(b[b.side == "O"].pivot_table(index="feature", columns="window", values="beta").round(2).to_string())
    print("\nbase beta, defense (flipped: positive = good):")
    print(b[b.side == "D"].pivot_table(index="feature", columns="window", values="beta").round(2).to_string())
