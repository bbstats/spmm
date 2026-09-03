"""Figures: one interactive Plotly page with metric tabs, plus static PNGs.

Colour carries the entity and nothing else: blue = offense, orange = defense, in both
light and dark steps (validated: worst adjacent CVD dE 24.7 light / 26.8 dark).  Every
other distinction -- base, rolling, lambda x3 / div 3, playoffs -- is line style, so a
filter never repaints a series.  Units everywhere are points per 100 possessions per unit
of the per-100 rate, defense flipped so positive = good.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- design tokens
LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8985",
             grid="#e6e5e1", zero="#c9c8c3", o="#2a78d6", d="#eb6834")
DARK = dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#8a8985",
            grid="#333331", zero="#4a4a47", o="#3987e5", d="#d95926")
ERA_STEPS = ["#86b6ef", "#2a78d6", "#0d366b"]          # ordinal blue: three era thirds
DIVERGE = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f3a3a2", "#e34948", "#8f1f1e"]
FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
LABEL = {"fg3m": "3PM", "fg3_miss": "3P miss", "fg2m": "2PM", "fg2_miss": "2P miss", "ftm": "FTM",
         "ft_miss": "FT miss", "orb": "ORB", "drb": "DRB", "ast": "AST", "tov": "TOV",
         "stl": "STL", "blk": "BLK", "pf": "PF"}
ORDER = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
BLURB = {
    "fg3m": "Low and uncertain in 1997-99, then steady near one point per 100 for the rest of the era.",
    "fg3_miss": "Went from mildly positive to costly as three-point volume climbed.",
    "fg2m": "Flat across thirty years on offense; the defensive side drifted to zero.",
    "fg2_miss": "Costs about half a point per 100 throughout, with no clear trend.",
    "ftm": "The steadiest coefficient in the set, near 0.6 in every window.",
    "ft_miss": "Noisy, and more costly late in the era than early.",
    "orb": "The largest mover: on defense it changed sign, from -0.29 to +0.24.",
    "drb": "Pays on the defensive side, not the offensive one, and slightly less than it used to.",
    "ast": "Near 0.4 on offense in every window.",
    "tov": "Costs about 0.7 points per 100 on both sides, flat across eras.",
    "stl": "The largest defensive coefficient in the model, around 1.5.",
    "blk": "Predicts defense more strongly late in the era, and offense less.",
    "pf": "Costs about 0.3 on offense; the defensive side sits near zero.",
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def load(cfg) -> dict:
    out = Path(cfg["_root"]) / cfg["paths"]["outputs"]
    d = dict(coefs=pd.read_parquet(out / "coefs.parquet"))
    for name in ("coefs_playoffs", "feature_corr", "variance"):
        p = out / f"{name}.parquet"
        if p.exists():
            d[name] = pd.read_parquet(p)
    return d


# ---------------------------------------------------------------- interactive figure
def interactive_coefs(data: dict, path: Path, cfg=None) -> Path:
    """One Plotly page: tab buttons pick the metric, traces show O and D across the windows."""
    import plotly.graph_objects as go

    c = data["coefs"]
    mid = c[c.run == "base"].drop_duplicates("window").set_index("window").window_mid
    po = data.get("coefs_playoffs")
    if po is not None:
        po = po[(po["pool"] == "window") & (po["lam_delta"] == po["lam_delta"].min())]

    fig = go.Figure()
    state: list = []                 # default visibility of each trace inside its own tab
    idx = {"O": [], "D": [], "ribbon_O": [], "ribbon_D": [], "ribbon_faint_O": [], "ribbon_faint_D": []}
    spans = []

    def add(trace, side, default=True, ribbon=None):
        fig.add_trace(trace)
        state.append(default)
        i = len(fig.data) - 1
        if ribbon is None:
            idx[side].append(i)
        else:
            idx[f"ribbon_{'faint_' if ribbon == 'faint' else ''}{side}"].append(i)

    for feat in ORDER:
        first = len(fig.data)
        cf = c[c.feature == feat]
        for side, key, nm in (("O", "o", "Offense"), ("D", "d", "Defense")):
            col = LIGHT[key]
            g = cf[(cf.side == side) & (cf.run == "base")].sort_values("window_mid")
            grp = f"{side}_base"
            for bound, fill in ((g.beta - 1.96 * g.se, None), (g.beta + 1.96 * g.se, "tonexty")):
                add(go.Scatter(x=g.window_mid, y=bound, mode="lines", line=dict(width=0), fill=fill,
                               fillcolor=_rgba(col, 0.16), hoverinfo="skip", showlegend=False, legendgroup=grp),
                    side, True, ribbon="solid")
            add(go.Scatter(x=g.window_mid, y=g.beta, name=nm, legendgroup=grp, mode="lines+markers",
                           line=dict(color=col, width=2),
                           marker=dict(size=9, color=col, line=dict(width=2, color=LIGHT["surface"])),
                           customdata=np.stack([g.window, g.se], axis=-1),
                           hovertemplate="<b>%{customdata[0]}</b><br>" + nm +
                                         ": %{y:.3f} ± %{customdata[1]:.3f}<extra></extra>"), side)

            r = cf[(cf.side == side) & (cf.run == "rolling")].sort_values("window_mid")
            add(go.Scatter(x=r.window_mid, y=r.beta, name=f"{nm}, rolling 3-yr", legendgroup=f"{side}_roll",
                           mode="lines", line=dict(color=_rgba(col, 0.45), width=1.5), hoverinfo="skip"),
                side, "legendonly")

            for k, (run, dash) in enumerate((("lam_x3", "dash"), ("lam_div3", "dot"))):
                q = cf[(cf.side == side) & (cf.run == run)].sort_values("window_mid")
                add(go.Scatter(x=q.window_mid, y=q.beta, name=f"{nm}, λ×3 / ÷3",
                               legendgroup=f"{side}_lam", showlegend=(k == 0), mode="lines",
                               line=dict(color=_rgba(col, 0.8), width=1, dash=dash), hoverinfo="skip"),
                    side, "legendonly")

            if po is not None and len(po):
                p_ = po[(po.side == side) & (po.feature == feat)].copy()
                p_["x"] = p_.window.map(mid)
                p_ = p_.sort_values("x")
                grp = f"{side}_po"
                for bound, fill in ((p_.po_beta - 1.96 * p_.po_se, None), (p_.po_beta + 1.96 * p_.po_se, "tonexty")):
                    add(go.Scatter(x=p_.x, y=bound, mode="lines", line=dict(width=0), fill=fill,
                                   fillcolor=_rgba(col, 0.10), hoverinfo="skip", showlegend=False, legendgroup=grp),
                        side, "legendonly", ribbon="faint")
                add(go.Scatter(x=p_.x, y=p_.po_beta, name=f"{nm}, playoffs", legendgroup=grp, mode="lines+markers",
                               line=dict(color=col, width=2, dash="dot"),
                               marker=dict(size=7, symbol="diamond", color=col),
                               customdata=np.stack([p_.window, p_.po_se], axis=-1),
                               hovertemplate="<b>%{customdata[0]}</b> playoffs<br>" + nm +
                                             ": %{y:.3f} ± %{customdata[1]:.3f}<extra></extra>"),
                    side, "legendonly")
        spans.append((first, len(fig.data)))

    n = len(fig.data)
    lo0, hi0 = spans[0]
    for i in range(n):
        fig.data[i].visible = state[i] if lo0 <= i < hi0 else False

    buttons = []
    for feat, (lo, hi) in zip(ORDER, spans):
        vis = [state[i] if lo <= i < hi else False for i in range(n)]
        buttons.append(dict(label=LABEL[feat], method="update", args=[
            {"visible": vis},
            {"yaxis.autorange": True,
             "title.text": f"{LABEL[feat]}: what one unit per 100 possessions is worth",
             "annotations[0].text": BLURB[feat]}]))

    fig.update_layout(
        template="none", font=dict(family=FONT, size=13, color=LIGHT["text"]),
        paper_bgcolor=LIGHT["surface"], plot_bgcolor=LIGHT["surface"],
        title=dict(text=f"{LABEL[ORDER[0]]}: what one unit per 100 possessions is worth",
                   font=dict(size=19, color=LIGHT["text"]), x=0.045, y=0.975, xanchor="left"),
        margin=dict(l=70, r=190, t=118, b=60), height=560, hovermode="x unified",
        xaxis=dict(title="window midpoint (season ending)", gridcolor=LIGHT["grid"], linecolor=LIGHT["grid"],
                   ticks="outside", tickcolor=LIGHT["grid"], tickfont=dict(color=LIGHT["text2"]),
                   title_font=dict(color=LIGHT["text2"]), zeroline=False, dtick=5),
        yaxis=dict(title="points per 100 possessions", gridcolor=LIGHT["grid"], linecolor=LIGHT["grid"],
                   ticks="outside", tickcolor=LIGHT["grid"], tickfont=dict(color=LIGHT["text2"]),
                   title_font=dict(color=LIGHT["text2"]), zeroline=True, zerolinecolor=LIGHT["zero"],
                   zerolinewidth=1),
        legend=dict(font=dict(color=LIGHT["text2"], size=12), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    x=1.01, y=1, yanchor="top", tracegroupgap=8),
        updatemenus=[dict(type="buttons", direction="right", showactive=True, active=0,
                          x=0.045, xanchor="left", y=1.155, yanchor="top",
                          bgcolor=LIGHT["surface"], bordercolor=LIGHT["grid"], borderwidth=1,
                          font=dict(size=12, color=LIGHT["text2"]), buttons=buttons)],
        annotations=[dict(text=BLURB[ORDER[0]], x=0.045, xref="paper", y=1.045, yref="paper", xanchor="left",
                          showarrow=False, font=dict(size=13, color=LIGHT["text2"]))],
    )
    html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="coefs",
                       config=dict(displaylogo=False, responsive=True,
                                   modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_page(html, idx), encoding="utf-8")
    return path


def _page(plot_html: str, idx: dict) -> str:
    import json
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA box-score coefficients across eras</title>
<style>
  :root {{ color-scheme: light; --surface:{LIGHT['surface']}; --text:{LIGHT['text']};
           --text2:{LIGHT['text2']}; --grid:{LIGHT['grid']}; --o:{LIGHT['o']}; --d:{LIGHT['d']}; }}
  :root[data-theme="dark"] {{ color-scheme: dark; --surface:{DARK['surface']}; --text:{DARK['text']};
           --text2:{DARK['text2']}; --grid:{DARK['grid']}; --o:{DARK['o']}; --d:{DARK['d']}; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ color-scheme: dark; --surface:{DARK['surface']}; --text:{DARK['text']};
           --text2:{DARK['text2']}; --grid:{DARK['grid']}; --o:{DARK['o']}; --d:{DARK['d']}; }} }}
  body {{ margin:0; background:var(--surface); color:var(--text); font-family:{FONT};
          -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:32px 24px 56px; }}
  h1 {{ font-size:26px; line-height:1.25; margin:0 0 10px; font-weight:600; letter-spacing:-0.015em; }}
  p.sub {{ color:var(--text2); font-size:15px; line-height:1.55; margin:0 0 4px; max-width:72ch; }}
  .key {{ display:flex; gap:20px; align-items:center; flex-wrap:wrap; margin:20px 0 4px;
          color:var(--text2); font-size:13px; }}
  .key span {{ display:inline-flex; align-items:center; gap:7px; }}
  .dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; }}
  .bar {{ display:inline-block; width:22px; height:0; }}
  footer {{ color:var(--text2); font-size:13.5px; line-height:1.6; margin-top:10px; max-width:78ch; }}
  footer p {{ margin:0 0 10px; }}
  a {{ color:var(--o); }}
  button.theme {{ float:right; background:transparent; color:var(--text2); border:1px solid var(--grid);
      border-radius:7px; padding:6px 12px; font-size:12px; cursor:pointer; font-family:inherit; }}
  button.theme:hover {{ color:var(--text); }}
  details {{ margin-top:26px; }}
  summary {{ cursor:pointer; color:var(--text2); font-size:13.5px; margin-bottom:10px; }}
</style></head><body>
<div class="wrap">
<button class="theme" id="themeBtn" aria-label="Toggle dark mode">Dark</button>
<h1>What the box score has been worth, 1997&ndash;2026</h1>
<p class="sub">Each coefficient is the points per 100 possessions that one unit per 100 of that
box-score stat is worth, holding the other twelve fixed. Estimated per three-season window by
mixed-model RAPM: the box score enters as unpenalized fixed effects beside ridge-penalized
player-season offense and defense effects, fit jointly. The defensive side is flipped so positive
is good on both. Bands are 95%.</p>
<div class="key">
  <span><i class="dot" style="background:var(--o)"></i> Offense</span>
  <span><i class="dot" style="background:var(--d)"></i> Defense</span>
  <span><i class="bar" style="border-top:2px solid var(--text2)"></i> base</span>
  <span><i class="bar" style="border-top:2px dotted var(--text2)"></i> playoffs</span>
  <span>pick a stat above; use the legend for rolling windows, &lambda;&times;3 / &divide;3 and playoffs</span>
</div>
{plot_html}
<details open><summary>How to read it</summary>
<footer>
<p><b>Held equal is doing work.</b> Three-point makes and misses correlate at 0.96 across players,
so nobody moves one without the other. A coefficient is the value of that stat given the rest of a
player's profile, not the value of a shot in isolation.</p>
<p><b>Defense means the defender's own rate.</b> &ldquo;BLK, defense&rdquo; is a player's own block
rate predicting how well his team defends with him on the floor, not blocks he allows.</p>
<p><b>The prior never sees the stints it is fit on.</b> Each season's games are split in half, and
the coefficients fit on one half use box-score rates computed only from the other. Full-season rates
inflate the three-point coefficient by more than seven standard errors, so this matters.</p>
<p><b>Standard errors are the model's</b>, and about 10% conservative against a game-level Bayesian
bootstrap. &lambda; is fixed across every window so it cannot manufacture drift; &lambda;&times;3 and
&divide;3 show what a threefold change in shrinkage would do.</p>
<p>Data, code and the coefficient tables as CSV:
<a href="https://github.com/bbstats/spmm">github.com/bbstats/spmm</a>.</p>
</footer></details>
</div>
<script>
const IDX = {json.dumps(idx)};
const L = {json.dumps(LIGHT)}, D = {json.dumps(DARK)};
function rgba(hex, a) {{
  const h = hex.slice(1);
  return `rgba(${{parseInt(h.slice(0,2),16)}},${{parseInt(h.slice(2,4),16)}},${{parseInt(h.slice(4,6),16)}},${{a}})`;
}}
function paint(mode) {{
  const t = mode === "dark" ? D : L, gd = document.getElementById("coefs");
  if (!gd || !window.Plotly) return;
  Plotly.relayout(gd, {{
    "paper_bgcolor": t.surface, "plot_bgcolor": t.surface, "font.color": t.text,
    "title.font.color": t.text, "xaxis.gridcolor": t.grid, "yaxis.gridcolor": t.grid,
    "xaxis.linecolor": t.grid, "yaxis.linecolor": t.grid, "xaxis.tickcolor": t.grid,
    "yaxis.tickcolor": t.grid, "xaxis.tickfont.color": t.text2, "yaxis.tickfont.color": t.text2,
    "xaxis.title.font.color": t.text2, "yaxis.title.font.color": t.text2,
    "yaxis.zerolinecolor": t.zero, "legend.font.color": t.text2,
    "updatemenus[0].bgcolor": t.surface, "updatemenus[0].bordercolor": t.grid,
    "updatemenus[0].font.color": t.text2, "annotations[0].font.color": t.text2
  }});
  for (const [side, key] of [["O","o"],["D","d"]]) {{
    Plotly.restyle(gd, {{"line.color": t[key], "marker.color": t[key],
                         "marker.line.color": t.surface}}, IDX[side]);
    Plotly.restyle(gd, {{"fillcolor": rgba(t[key], 0.16)}}, IDX["ribbon_" + side]);
    Plotly.restyle(gd, {{"fillcolor": rgba(t[key], 0.10)}}, IDX["ribbon_faint_" + side]);
  }}
}}
function whenReady(fn) {{
  const gd = document.getElementById("coefs");
  if (window.Plotly && gd && gd._fullLayout) fn();
  else setTimeout(() => whenReady(fn), 60);
}}
const btn = document.getElementById("themeBtn");
function setTheme(m) {{
  document.documentElement.dataset.theme = m;
  btn.textContent = m === "dark" ? "Light" : "Dark";
  whenReady(() => paint(m));
}}
btn.addEventListener("click", () =>
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
if (window.matchMedia("(prefers-color-scheme: dark)").matches) setTheme("dark");
</script>
</body></html>"""


# ---------------------------------------------------------------- static figures
def _mpl(dark=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = DARK if dark else LIGHT
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["text"], "axes.labelcolor": t["text2"], "xtick.color": t["text2"],
        "ytick.color": t["text2"], "axes.edgecolor": t["grid"], "grid.color": t["grid"], "font.size": 11,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    return plt, t


def metric_pngs(data: dict, out_dir: Path) -> list:
    """One PNG per metric: offense and defense across the ten windows, 95% bands."""
    plt, t = _mpl()
    out_dir.mkdir(parents=True, exist_ok=True)
    b = data["coefs"][data["coefs"].run == "base"]
    paths = []
    for feat in ORDER:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for side, key, nm in (("O", "o", "Offense"), ("D", "d", "Defense")):
            g = b[(b.feature == feat) & (b.side == side)].sort_values("window_mid")
            ax.fill_between(g.window_mid, g.beta - 1.96 * g.se, g.beta + 1.96 * g.se,
                            color=t[key], alpha=0.16, lw=0)
            ax.plot(g.window_mid, g.beta, color=t[key], lw=2, marker="o", ms=5.5,
                    mec=t["surface"], mew=1.5, zorder=3)
            ax.annotate(nm, (g.window_mid.iloc[-1], g.beta.iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", color=t["text"], fontsize=11, va="center", fontweight=500)
        ax.axhline(0, color=t["zero"], lw=1, zorder=1)
        ax.grid(axis="y", lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_xlabel("window midpoint (season ending)")
        ax.set_ylabel("points per 100 possessions")
        ax.set_title(f"{LABEL[feat]}: what one unit per 100 possessions is worth",
                     color=t["text"], fontsize=13.5, loc="left", pad=30, fontweight="semibold")
        ax.text(0, 1.035, BLURB[feat], transform=ax.transAxes, color=t["text2"], fontsize=10.5, va="bottom")
        xs = b[b.feature == feat].window_mid
        ax.set_xlim(xs.min() - 1.2, xs.max() + 3.4)      # room on the right for the direct labels
        fig.tight_layout()
        p = out_dir / f"coef_{feat}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def playoff_delta_png(data: dict, path: Path) -> Path:
    """Dot-and-whisker of delta (playoff minus regular season) by era third, 13 metrics x O/D."""
    plt, t = _mpl()
    po = data["coefs_playoffs"]
    d = po[(po["pool"] == "era_third") & (po["lam_delta"] == po["lam_delta"].min())]
    thirds = sorted(d.window.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.6), sharey=True, sharex=True)
    ypos = {f: len(ORDER) - 1 - i for i, f in enumerate(ORDER)}
    for ax, side, nm in zip(axes, ("O", "D"), ("Offense", "Defense")):
        ax.axvline(0, color=t["zero"], lw=1.2, zorder=1)
        for k, th in enumerate(thirds):
            off = (1 - k) * 0.26          # oldest era on top, reading chronologically downward
            g = d[(d.side == side) & (d.window == th)]
            for _, r in g.iterrows():
                y = ypos[r.feature] + off
                ax.plot([r.delta - 1.96 * r.delta_se, r.delta + 1.96 * r.delta_se], [y, y],
                        color=ERA_STEPS[k], lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot([r.delta], [y], "o", color=ERA_STEPS[k], ms=6, mec=t["surface"], mew=1.2, zorder=3,
                        label=th if (r.feature == ORDER[0] and side == "O") else None)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([LABEL[f] for f in reversed(ORDER)])
        ax.grid(axis="x", lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_title(nm, color=t["text"], fontsize=12.5, loc="left", pad=8, fontweight="semibold")
        ax.set_xlabel("playoff minus regular season, points per 100")
    for ax in axes:
        lim = max(abs(np.array(ax.get_xlim()))) * 1.04
        ax.set_xlim(-lim, lim)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, title="seasons", frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.055), fontsize=10, title_fontsize=10, labelcolor=t["text2"],
               columnspacing=2.4, handletextpad=0.5)
    fig.suptitle("What the box score is worth in the playoffs, minus the regular season",
                 x=0.008, y=1.07, ha="left", color=t["text"], fontsize=15, fontweight="semibold")
    fig.text(0.008, 1.005, "Positive = worth more in the playoffs. Cross-validation prefers shrinking every "
                           "one of these to zero; only three of 26 clear two standard errors.",
             ha="left", color=t["text2"], fontsize=10.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.965])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def correlation_pngs(data: dict, out_dir: Path, windows=("1997-1999", "2024-2026")) -> list:
    """Possession-weighted correlation of the 13 rates: first and last window, and the change."""
    from matplotlib.colors import LinearSegmentedColormap
    plt, t = _mpl()
    fc = data["feature_corr"]
    cmap = LinearSegmentedColormap.from_list("bl_rd", DIVERGE[::-1], N=256)
    out_dir.mkdir(parents=True, exist_ok=True)

    def mat(win):
        g = fc[fc.window == win]
        m = g.pivot(index="feature_i", columns="feature_j", values="corr").reindex(index=ORDER, columns=ORDER)
        return m.to_numpy()

    ms = [mat(w) for w in windows]
    panels = [(windows[0], ms[0]), (windows[1], ms[1]),
              (f"change, {windows[0]} to {windows[1]}", ms[1] - ms[0])]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    labels = [LABEL[f] for f in ORDER]
    for ax, (title, m) in zip(axes, panels):
        lim = 1.0 if "change" not in title else 0.6
        im = ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(labels, rotation=90, fontsize=9)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, color=t["text"], fontsize=12, loc="left", pad=8, fontweight="semibold")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9, length=0, colors=t["text2"])
    fig.suptitle("Who does what, together: correlation of the 13 per-100 rates across player-seasons",
                 x=0.005, y=1.05, ha="left", color=t["text"], fontsize=14.5, fontweight="semibold")
    fig.text(0.005, 1.0, "A coefficient can move because the value of the stat changed, or because the "
                         "players who accumulate it changed. This panel is the second one.",
             ha="left", color=t["text2"], fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out_dir / "correlations.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return [p]


def movers_png(data: dict, path: Path, top=10) -> Path:
    """Biggest first-to-last movers, as a dot-and-whisker of the change in each coefficient."""
    plt, t = _mpl()
    b = data["coefs"][data["coefs"].run == "base"]
    rows = []
    for (s, f), g in b.groupby(["side", "feature"]):
        g = g.sort_values("window_mid")
        ch = g.beta.iloc[-1] - g.beta.iloc[0]
        se = float(np.hypot(g.se.iloc[0], g.se.iloc[-1]))
        rows.append(dict(side=s, feature=f, change=ch, se=se, z=ch / se))
    df = pd.DataFrame(rows)
    m = df.reindex(df.z.abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(m) + 1.7))
    ax.axvline(0, color=t["zero"], lw=1.2)
    for i, (_, r) in enumerate(m.iterrows()):
        col = t["o"] if r.side == "O" else t["d"]
        ax.plot([r.change - 1.96 * r.se, r.change + 1.96 * r.se], [i, i], color=col, lw=1.8,
                solid_capstyle="round")
        ax.plot([r.change], [i], "o", color=col, ms=7, mec=t["surface"], mew=1.4, zorder=3)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f"{LABEL[r.feature]} ({'offense' if r.side == 'O' else 'defense'})"
                        for _, r in m.iterrows()])
    ax.grid(axis="x", lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_xlabel("change from 1997-99 to 2024-26, points per 100")
    ax.set_title("What moved most", color=t["text"], fontsize=13.5, loc="left", pad=30, fontweight="semibold")
    ax.text(0, 1.03, "Blue = offense, orange = defense. Bars are 95% intervals on the change.",
            transform=ax.transAxes, color=t["text2"], fontsize=10.5, va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- landing page
def index_page(path: Path, figs: list) -> Path:
    """A small landing page for GitHub Pages: the interactive figure plus the static ones."""
    cards = "\n".join(
        f'<figure><img src="img/{name}" alt="{alt}" loading="lazy"><figcaption>{cap}</figcaption></figure>'
        for name, alt, cap in figs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA box-score coefficients across eras</title>
<meta name="description" content="How the mapping from box score to on-court impact changed from 1996-97 to 2025-26, estimated by mixed-model RAPM.">
<style>
  :root {{ color-scheme: light; --surface:{LIGHT['surface']}; --text:{LIGHT['text']};
           --text2:{LIGHT['text2']}; --grid:{LIGHT['grid']}; --o:{LIGHT['o']}; --d:{LIGHT['d']}; }}
  :root[data-theme="dark"] {{ color-scheme: dark; --surface:{DARK['surface']}; --text:{DARK['text']};
           --text2:{DARK['text2']}; --grid:{DARK['grid']}; --o:{DARK['o']}; --d:{DARK['d']}; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark;
           --surface:{DARK['surface']}; --text:{DARK['text']}; --text2:{DARK['text2']};
           --grid:{DARK['grid']}; --o:{DARK['o']}; --d:{DARK['d']}; }} }}
  body {{ margin:0; background:var(--surface); color:var(--text); font-family:{FONT};
          -webkit-font-smoothing:antialiased; line-height:1.55; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:40px 24px 72px; }}
  h1 {{ font-size:30px; line-height:1.2; margin:0 0 10px; font-weight:600; letter-spacing:-0.015em; }}
  h2 {{ font-size:19px; margin:44px 0 6px; font-weight:600; letter-spacing:-0.01em; }}
  p {{ color:var(--text2); font-size:15.5px; max-width:70ch; margin:0 0 12px; }}
  a {{ color:var(--o); }}
  .cta {{ display:inline-block; margin:18px 0 6px; padding:11px 18px; border:1px solid var(--grid);
          border-radius:9px; text-decoration:none; color:var(--text); font-size:15px; font-weight:500; }}
  .cta:hover {{ border-color:var(--o); }}
  figure {{ margin:26px 0 8px; }}
  img {{ width:100%; height:auto; border-radius:8px; }}
  figcaption {{ color:var(--text2); font-size:13.5px; margin-top:8px; }}
  ul {{ color:var(--text2); font-size:15px; max-width:70ch; padding-left:20px; }}
  li {{ margin-bottom:8px; }}
  button.theme {{ float:right; background:transparent; color:var(--text2); border:1px solid var(--grid);
      border-radius:7px; padding:6px 12px; font-size:12px; cursor:pointer; font-family:inherit; }}
</style></head><body><div class="wrap">
<button class="theme" id="t">Dark</button>
<h1>What the box score has been worth, 1997&ndash;2026</h1>
<p>Thirty seasons of play-by-play, turned into stints, turned into one ridge regression per
three-season window. The box score enters as unpenalized fixed effects beside ridge-penalized
player-season offense and defense effects, fit jointly by Henderson's mixed model equations, so the
coefficients land on the same scale as on-court impact: points per 100 possessions per unit per 100.
The defensive side is flipped so positive is good on both.</p>
<a class="cta" href="coefs.html">Open the interactive figure &rarr;</a>
{cards}
<h2>Method, briefly</h2>
<ul>
<li><b>Data.</b> Every regular-season and playoff game from 1996-97 to 2025-26, reconstructed from
stats.nba.com play-by-play into possessions with the ten players on the floor. Points reconcile to
the box score in every game, and 99.9% of possessions have five players a side.</li>
<li><b>Cross-fitted rates.</b> Each season's games are split in half. The coefficients fit on one
half use box-score rates computed only from the other, so a coefficient never sees the box score of
the stints it is fit on. Using full-season rates inflates the three-point coefficient by more than
seven standard errors.</li>
<li><b>One &lambda; for the whole era</b>, chosen once on two windows by cross-validation and REML,
so shrinkage cannot manufacture drift. &lambda;&times;3 and &divide;3 are on the figure.</li>
<li><b>Standard errors</b> are the model's, about 10% conservative against a game-level Bayesian
bootstrap.</li>
</ul>
<h2>Everything else</h2>
<p>Code, coefficient tables as CSV, per-season data diagnostics and the checkpoint logs:
<a href="https://github.com/bbstats/spmm">github.com/bbstats/spmm</a>.</p>
<script>
const b=document.getElementById("t");
function set(m) {{ document.documentElement.dataset.theme=m; b.textContent = m==="dark" ? "Light" : "Dark"; }}
b.onclick=() => set(document.documentElement.dataset.theme==="dark" ? "light" : "dark");
if (matchMedia("(prefers-color-scheme: dark)").matches) set("dark");
</script>
</div></body></html>""", encoding="utf-8")
    return path
