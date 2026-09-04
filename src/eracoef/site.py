"""One page for GitHub Pages: the player-ratings leaderboard on top, the coefficients under it.

The leaderboard is drawn client-side from an embedded JSON blob rather than as a pre-built Plotly
figure, because it is a two-dimensional control (which window x which view) and baking every
combination into trace-visibility masks would be both larger and harder to read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .plots import DARK, FONT, LIGHT, interactive_coefs

TOP_N = 25
MISS_N = 11
MIN_POSS = 1000
MIN_POSS_MISS = 3000


def leaderboard_data(rat: pd.DataFrame) -> dict:
    """Per window: the top players by rating, and the ones the box score most under- and over-rates."""
    cols = ["player_name", "poss_off", "prior_total", "boost_total", "u_total",
            "rating_off", "rating_def", "rating_total"]
    out = {}
    for w, d in rat.groupby("window"):
        d = d[d.player_name.notna()].copy()
        top = d[d.poss_off >= MIN_POSS].nlargest(TOP_N, "rating_total")
        m = d[d.poss_off >= MIN_POSS_MISS]
        miss = pd.concat([m.nlargest(MISS_N, "u_total"), m.nsmallest(MISS_N, "u_total")])
        miss = miss.sort_values("u_total", ascending=False)
        out[w] = {k: [{c: (r[c] if c == "player_name" else round(float(r[c]), 3)) for c in cols}
                      for _, r in v.iterrows()] for k, v in (("top", top), ("miss", miss))}
    return out


def build(cfg, ratings: pd.DataFrame, data: dict, figs: list, path: Path) -> Path:
    """Write the single page: leaderboard, coefficient figure, static figures, method."""
    plot_html, idx = interactive_coefs(data, cfg=cfg)
    board = leaderboard_data(ratings)
    wins = sorted(board)
    cards = "\n".join(
        f'<figure><img src="img/{name}" alt="{alt}" loading="lazy"><figcaption>{cap}</figcaption></figure>'
        for name, alt, cap in figs)
    css = _CSS
    # Longest key first: "text" is a prefix of "text2", so a naive dict-order pass rewrites the
    # "$L_text" inside "$L_text2" and leaves a stray digit behind, e.g. "--text2:#0b0b0b2".
    for prefix, theme in (("$L_", LIGHT), ("$D_", DARK)):
        for k in sorted(theme, key=len, reverse=True):
            css = css.replace(prefix + k, theme[k])
    css = css.replace("$FONT", FONT)
    buttons = "\n".join(
        f'<button class="win{" on" if w == wins[-1] else ""}" data-w="{w}">{w}</button>' for w in wins)
    html = (_HTML
            .replace("__CSS__", css)
            .replace("__BOARD__", json.dumps(board, ensure_ascii=True))
            .replace("__WINS__", json.dumps(wins))
            .replace("__LIGHT__", json.dumps(LIGHT))
            .replace("__DARK__", json.dumps(DARK))
            .replace("__FONT__", json.dumps(FONT))
            .replace("__BUTTONS__", buttons)
            .replace("__PLOT__", plot_html)
            .replace("__IDXJSON__", json.dumps({k: list(map(int, v)) for k, v in idx.items()}))
            .replace("__CARDS__", cards)
            .replace("__LATEST__", wins[-1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_CSS = """
  :root { color-scheme: light; --surface:$L_surface; --text:$L_text; --text2:$L_text2;
          --muted:$L_muted; --grid:$L_grid; --zero:$L_zero; --t:$L_t; --o:$L_o; --d:$L_d; }
  :root[data-theme="dark"] { color-scheme: dark; --surface:$D_surface; --text:$D_text;
          --text2:$D_text2; --muted:$D_muted; --grid:$D_grid; --zero:$D_zero;
          --t:$D_t; --o:$D_o; --d:$D_d; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { color-scheme: dark; --surface:$D_surface; --text:$D_text;
          --text2:$D_text2; --muted:$D_muted; --grid:$D_grid; --zero:$D_zero;
          --t:$D_t; --o:$D_o; --d:$D_d; } }
  body { margin:0; background:var(--surface); color:var(--text); font-family:$FONT;
         -webkit-font-smoothing:antialiased; line-height:1.55; }
  .wrap { max-width:1120px; margin:0 auto; padding:36px 24px 72px; }
  h1 { font-size:31px; line-height:1.2; margin:0 0 12px; font-weight:600; letter-spacing:-0.015em; }
  h2 { font-size:21px; margin:56px 0 8px; font-weight:600; letter-spacing:-0.012em; }
  p { color:var(--text2); font-size:15.5px; max-width:72ch; margin:0 0 12px; }
  a { color:var(--t); }
  figure { margin:26px 0 8px; }
  img { width:100%; height:auto; border-radius:8px; }
  figcaption { color:var(--text2); font-size:13.5px; margin-top:8px; max-width:72ch; }
  ul { color:var(--text2); font-size:15px; max-width:72ch; padding-left:20px; }
  li { margin-bottom:8px; }
  .controls { display:flex; gap:22px; align-items:center; flex-wrap:wrap; margin:22px 0 6px; }
  .grp { display:flex; gap:6px; flex-wrap:wrap; }
  .grp button { background:transparent; color:var(--text2); border:1px solid var(--grid);
        border-radius:7px; padding:6px 11px; font-size:12.5px; cursor:pointer; font-family:inherit;
        line-height:1.2; }
  .grp button:hover { color:var(--text); border-color:var(--muted); }
  .grp button.on { color:var(--surface); background:var(--t); border-color:var(--t); font-weight:500; }
  .key { display:flex; gap:18px; align-items:center; flex-wrap:wrap; color:var(--text2);
         font-size:13px; margin:14px 0 2px; }
  .key span { display:inline-flex; align-items:center; gap:7px; }
  .dot { width:11px; height:11px; border-radius:3px; display:inline-block; }
  button.theme { float:right; background:transparent; color:var(--text2); border:1px solid var(--grid);
        border-radius:7px; padding:6px 12px; font-size:12px; cursor:pointer; font-family:inherit; }
  footer { color:var(--text2); font-size:13.5px; line-height:1.6; margin-top:34px; max-width:78ch; }
  details { margin-top:20px; }
  summary { cursor:pointer; color:var(--text2); font-size:13.5px; }
"""

_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA player ratings and box-score coefficients, 1997-2026</title>
<meta name="description" content="Mixed-model RAPM player ratings and how the mapping from box score to on-court impact changed from 1996-97 to 2025-26.">
<style>__CSS__</style>
</head><body><div class="wrap">
<button class="theme" id="themeBtn" aria-label="Toggle dark mode">Dark</button>
<h1>Thirty years of NBA players, rated by what the box score knows and what it misses</h1>
<p>One ridge regression per three-season window. The box score enters as unpenalized fixed effects
beside ridge-penalized player offense and defense effects, fit jointly by Henderson's mixed model
equations, so a player's rating and the coefficients land on the same scale: points per 100
possessions. Every player gets one effect per three-year window. Defense is flipped so positive is
good on both sides.</p>

<h2>The leaderboard</h2>
<p>A rating is three things added together: what the box score says about you, a gradient-boosted
correction for where the box score's straight-line pricing breaks down, and what the on-court
margin says on top of both. Switch to <b>offense and defense</b> to see the same totals split by
side, or to <b>what the box score misses</b> for the players the margin disagrees with most.</p>
<div class="controls">
  <div class="grp" id="views">
    <button class="on" data-v="build">How a rating is built</button>
    <button data-v="sides">Offense and defense</button>
    <button data-v="miss">What the box score misses</button>
  </div>
</div>
<div class="controls"><div class="grp" id="wins">__BUTTONS__</div></div>
<div class="key" id="key"></div>
<div id="board"></div>
<figcaption id="cap"></figcaption>

<h2>What the box score has been worth</h2>
<p>Each coefficient is the points per 100 possessions that one unit per 100 of that stat is worth,
holding the other twelve fixed. The tabs pick the stat; bands are 95%.</p>
__PLOT__
__CARDS__

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
so shrinkage cannot manufacture drift.</li>
<li><b>The nonlinear correction</b> is a gradient-boosted model on the ridge residual, pooled over
all ten windows, cross-fitted so a player never scores himself, shrunk by the slope of its own
out-of-fold prediction, and stripped of any linear component so it can only supply curvature. On
held-out games it moves the calibration slope of the prior for low-minute players from 0.84 to 0.97
on defense while leaving starters unchanged.</li>
<li><b>Standard errors</b> are the model's, about 10% conservative against a game-level Bayesian
bootstrap.</li>
</ul>
<footer><p>Code, coefficient tables and player ratings as CSV, and the checkpoint logs:
<a href="https://github.com/bbstats/spmm">github.com/bbstats/spmm</a>.</p></footer>
</div>
<script>
const BOARD = __BOARD__, WINS = __WINS__, L = __LIGHT__, D = __DARK__, FONT = __FONT__;
window.__COEF_IDX__ = __IDXJSON__;
const VIEWS = {
  build: {rows:"top", keys:["prior_total","boost_total","u_total"], slots:["t","o","d"],
          names:["Box score","Nonlinear correction","On the floor"], unit:"rating",
          cap:"Top 25 by rating, minimum 1000 possessions. Each bar is the box score's opinion, "
             +"the correction for where its straight-line pricing breaks down, and what the "
             +"on-court margin adds on top of both."},
  sides: {rows:"top", keys:["rating_off","rating_def"], slots:["o","d"], names:["Offense","Defense"], unit:"rating",
          cap:"The same 25 players, split into offense and defense. Positive is good on both sides."},
  miss:  {rows:"miss", keys:["u_total"], slots:["t"], names:["On the floor, beyond the box score"], unit:"residual",
          cap:"The players whose on-court margin disagrees most with their box score, at 3000 or "
             +"more possessions. Positive means the floor likes him more than the box score does."}
};
let state = {win: "__LATEST__", view: "build"};

function theme() { return document.documentElement.dataset.theme === "dark" ? D : L; }
function fmt(v) { return (v >= 0 ? "+" : "\\u2212") + Math.abs(v).toFixed(2); }

function draw() {
  const t = theme(), V = VIEWS[state.view], rows = BOARD[state.win][V.rows];
  const names = rows.map(r => r.player_name), n = rows.length;
  const total = rows.map(r => V.keys.reduce((a, k) => a + r[k], 0));
  const traces = V.keys.map((k, i) => ({
    type:"bar", orientation:"h", name:V.names[i],
    y:names, x:rows.map(r => r[k]),
    marker:{color:t[V.slots[i]], line:{color:t.surface, width:1.5}},
    hovertemplate:"<b>%{y}</b><br>" + V.names[i] + ": %{x:+.2f}<extra></extra>"
  }));
  let lo = 0, hi = 0;
  rows.forEach((r, j) => {
    let pos = 0, neg = 0;
    V.keys.forEach(k => { if (r[k] > 0) pos += r[k]; else neg += r[k]; });
    hi = Math.max(hi, pos, total[j]); lo = Math.min(lo, neg, total[j]);
  });
  const pad = (hi - lo) * 0.17;
  const ann = rows.map((r, j) => ({
    x: hi + pad * 0.98, y: names[j], xref:"x", yref:"y", text: fmt(total[j]), showarrow:false,
    xanchor:"right", font:{size:12.5, color:t.text, family:FONT}
  }));
  ann.push({x: hi + pad * 0.98, y: 1.0, yref:"paper", xref:"x", text:V.unit, showarrow:false,
            xanchor:"right", yanchor:"bottom", font:{size:11.5, color:t.text2, family:FONT}});
  const layout = {
    barmode:"relative", bargap:0.30, height: 74 + 27 * n,
    margin:{l:174, r:14, t:24, b:46}, paper_bgcolor:t.surface, plot_bgcolor:t.surface,
    font:{family:FONT, color:t.text2, size:12.5}, showlegend:false, annotations:ann,
    hoverlabel:{bgcolor:t.surface, bordercolor:t.grid, font:{family:FONT, color:t.text}},
    xaxis:{title:{text:"points per 100 possessions", font:{color:t.text2, size:12}},
           range:[lo - pad * 0.10, hi + pad * 1.06], gridcolor:t.grid, linecolor:t.grid, ticks:"outside",
           tickcolor:t.grid, tickfont:{color:t.text2}, zeroline:true, zerolinecolor:t.zero,
           zerolinewidth:1},
    yaxis:{autorange:"reversed", gridcolor:"rgba(0,0,0,0)", linecolor:t.grid, ticks:"",
           tickfont:{color:t.text, size:12.5}, automargin:true}
  };
  Plotly.react("board", traces, layout,
               {displaylogo:false, responsive:true, displayModeBar:false});
  document.getElementById("key").innerHTML = V.keys.map((k, i) =>
    '<span><i class="dot" style="background:' + t[V.slots[i]] + '"></i>' + V.names[i] + "</span>").join("");
  document.getElementById("cap").textContent = V.cap;
}

document.getElementById("views").addEventListener("click", e => {
  if (!e.target.dataset.v) return;
  state.view = e.target.dataset.v;
  [...e.currentTarget.children].forEach(b => b.classList.toggle("on", b === e.target));
  draw();
});
document.getElementById("wins").addEventListener("click", e => {
  if (!e.target.dataset.w) return;
  state.win = e.target.dataset.w;
  [...e.currentTarget.children].forEach(b => b.classList.toggle("on", b === e.target));
  draw();
});

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return "rgba(" + (n >> 16 & 255) + "," + (n >> 8 & 255) + "," + (n & 255) + "," + a + ")";
}
function paintCoefs(mode) {
  const gd = document.getElementById("coefs"); if (!gd || !gd._fullLayout) return;
  const t = mode === "dark" ? D : L;
  Plotly.relayout(gd, {"paper_bgcolor":t.surface, "plot_bgcolor":t.surface,
    "font.color":t.text2, "xaxis.gridcolor":t.grid, "xaxis.linecolor":t.grid,
    "xaxis.tickcolor":t.grid, "xaxis.tickfont.color":t.text2, "xaxis.title.font.color":t.text2,
    "yaxis.gridcolor":t.grid, "yaxis.linecolor":t.grid, "yaxis.tickcolor":t.grid,
    "yaxis.tickfont.color":t.text2, "yaxis.title.font.color":t.text2, "yaxis.zerolinecolor":t.zero,
    "legend.font.color":t.text2, "updatemenus[0].bgcolor":t.surface,
    "updatemenus[0].bordercolor":t.grid, "updatemenus[0].font.color":t.text2,
    "annotations[0].font.color":t.text2,
    "hoverlabel.bgcolor":t.surface, "hoverlabel.bordercolor":t.grid, "hoverlabel.font.color":t.text});
  const idx = window.__COEF_IDX__; if (!idx) return;
  for (const pair of [["Total","t"],["O","o"],["D","d"]]) {
    const side = pair[0], key = pair[1];
    if (idx[side]) Plotly.restyle(gd, {"line.color":t[key], "marker.color":t[key],
                                       "marker.line.color":t.surface}, idx[side]);
    if (idx["ribbon_" + side]) Plotly.restyle(gd, {"fillcolor":rgba(t[key], 0.16)}, idx["ribbon_" + side]);
    if (idx["ribbon_faint_" + side]) Plotly.restyle(gd, {"fillcolor":rgba(t[key], 0.10)}, idx["ribbon_faint_" + side]);
  }
}
const btn = document.getElementById("themeBtn");
function setTheme(m) {
  document.documentElement.dataset.theme = m;
  btn.textContent = m === "dark" ? "Light" : "Dark";
  draw(); paintCoefs(m);
}
btn.addEventListener("click", function () {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});
if (window.matchMedia("(prefers-color-scheme: dark)").matches) { setTheme("dark"); }
else { draw(); }
</script>
</body></html>
"""
