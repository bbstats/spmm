"""Fit (or refit) the per-season shot-distance curves and gate them.

A curve is accepted when every one-foot bin with at least 1000 attempts reproduces its observed
make rate within 2 points.  The rim structure is sharper than any smooth curve (dunks at 0 ft,
layups at 1, contested shots at 2-3), which is why dense bins keep their own padded rate and the
spline serves only the sparse ones (shotcurve.K_BIN).

usage: python scripts/47_shotcurve_gate.py [first] [last] [--force]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eracoef.config import load_config  # noqa: E402
from eracoef.shotcurve import UNLOCATED, calibration, curve_for, curve_path, fit_curve, season_shot_rows  # noqa: E402

cfg = load_config()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(args[0]) if args else int(cfg["first_season"])
LAST = int(args[1]) if len(args) > 1 else int(cfg["last_season"])
FORCE = "--force" in sys.argv
t0 = time.time()
worst, bad_total = 0.0, 0
for s in range(FIRST, LAST + 1):
    rows = season_shot_rows(s, cfg)
    if FORCE or not curve_path(s, cfg).exists():
        c = fit_curve(rows, s)
        curve_path(s, cfg).parent.mkdir(parents=True, exist_ok=True)
        c.table.to_parquet(curve_path(s, cfg), index=False)
    else:
        c = curve_for(s, cfg)
    cal = calibration(c, rows, min_n=1000)
    bad = cal[cal.gap.abs() > 0.02]
    worst = max(worst, float(cal.gap.abs().max()))
    bad_total += len(bad)
    unloc3 = float((rows[rows.three].bin == UNLOCATED).mean())
    print(f"{s}: {len(rows)} attempts, bins>=1000 {len(cal)}, worst gap {cal.gap.abs().max():.4f}, over 2%: {len(bad)}  "
          f"rim0 {c.prob_bin(0, False):.3f} rim1 {c.prob_bin(1, False):.3f} 10ft {c.prob_bin(10, False):.3f} "
          f"3pt25 {c.prob_bin(25, True):.3f} unlocated threes {100 * unloc3:.1f}% ({time.time() - t0:.0f}s)", flush=True)
    if len(bad):
        print(bad.to_string(index=False))
print(f"GATE {'PASSED' if bad_total == 0 else 'FAILED'}: worst gap {worst:.4f}, bins over 2%: {bad_total}")
sys.exit(0 if bad_total == 0 else 1)
