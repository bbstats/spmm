"""Renders outputs/diagnostic.html from the payload scripts/21_diagnostic.py computes.

Figures are drawn client-side from one embedded blob, the same way src/eracoef/site.py does the
leaderboard, so the theme toggle redraws everything for free and every point keeps its hover.

Colour rule, held to across every figure on the page: a hue means a SIDE.
  blue = Total, orange = Offense, green = Defense
The only exception is figure 1, where the three things being compared are stages of one build-up
rather than sides, so it uses an ordinal light-to-dark blue ramp instead.

The light green sits below 3:1 against the light surface, so every figure carries direct value
labels or a table view.  Both are provided.
"""
from __future__ import annotations

import json
from pathlib import Path

_CSS = """
:root { color-scheme: light; --surface:$L_surface; --text:$L_text; --text2:$L_text2;
        --muted:$L_muted; --grid:$L_grid; --zero:$L_zero; --t:$L_t; --o:$L_o; --d:$L_d; }
:root[data-theme="dark"] { color-scheme: dark; --surface:$D_surface; --text:$D_text;
        --text2:$D_text2; --muted:$D_muted; --grid:$D_grid; --zero:$D_zero;
        --t:$D_t; --o:$D_o; --d:$D_d; }
* { box-sizing: border-box; }
body { margin:0; padding:32px 24px 72px; background:var(--surface); color:var(--text);
       font-family:$FONT; line-height:1.55; }
main { max-width:1080px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:19px; margin:44px 0 4px; letter-spacing:-0.01em; }
p { color:var(--text2); margin:6px 0 14px; font-size:14.5px; max-width:74ch; }
p.lede { font-size:15.5px; }
.sub { color:var(--muted); font-size:13px; }
.row { display:flex; gap:14px; flex-wrap:wrap; }
.row > div { flex:1 1 300px; min-width:280px; }
.plot { width:100%; min-height:260px; }
button { font:inherit; font-size:13px; color:var(--text2); background:transparent;
         border:1px solid var(--grid); border-radius:7px; padding:4px 11px; cursor:pointer; }
button:hover { border-color:var(--muted); color:var(--text); }
button.on { background:var(--text); color:var(--surface); border-color:var(--text); }
#theme { position:fixed; top:14px; right:16px; z-index:9; }
table { border-collapse:collapse; font-size:13px; margin:6px 0 0; width:100%; }
th,td { text-align:right; padding:4px 10px; border-bottom:1px solid var(--grid); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; }
.tbl { display:none; }
.tbl.on { display:block; }
.note { border-left:3px solid var(--grid); padding:2px 0 2px 14px; margin:14px 0; }
"""

_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ratings diagnostic</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>__CSS__</style></head>
<body data-palette="__PALETTE__">
<button id="theme">dark</button>
<main>
<h1>Ratings diagnostic</h1>
<p class="lede">Every figure below is scored against the same benchmark, and that benchmark
contains no box score at all: the same player&rsquo;s <b>next</b> three-year window, measured by
on-court margin only. Different games, different teammates, nothing the box score can leak into.
__NPAIRS__ players qualify by playing __MINPOSS__ possessions in two consecutive windows.
Units are points per 100 possessions unless a figure says otherwise.</p>

<h2>1. What a rating is actually made of</h2>
<p>The spread of each piece across the whole board. This is the honest version of &ldquo;how much of
this is box score&rdquo;, and the answer is <b>most of it</b>. The imbalance is worst on offense,
where the prior has seven times the spread of the on-court residual (2.15 against 0.30). On defense
the on-court part carries three times as much (0.95): the box score has less to say about defense,
so the margin fills more of the gap.</p>
<div id="f1" class="plot"></div>
<button class="tt" data-for="t1">show numbers</button><div id="t1" class="tbl"></div>

<h2>2. Does the shrinkage constant matter?</h2>
<p>How well the rating predicts the benchmark, as the shrinkage constant on the on-court part is
swept across three orders of magnitude. The curve is almost flat across the middle: the shipped
value and the best value differ by about seven thousandths of a correlation. This is the knob a
previous round of work expected to be decisive. It is not.</p>
<div id="f2" class="plot"></div>
<button class="tt" data-for="t2">show numbers</button><div id="t2" class="tbl"></div>

<h2>3. Does the board over-rate big men?</h2>
<p>Tilt toward bigs, measured per standard deviation of each thing so the three are comparable. If
the rating over-credited bigs, its bar would stand taller than the evidence bar. <b>On the total it
does not</b> &mdash; the on-court evidence likes bigs slightly more than the rating does. On each
side separately the rating is about twice as steep as the evidence, in opposite directions, and the
two cancel when they are added.</p>
<div class="row"><div id="f3t" class="plot"></div><div id="f3o" class="plot"></div>
<div id="f3d" class="plot"></div></div>
<button class="tt" data-for="t3">show numbers</button><div id="t3" class="tbl"></div>

<h2>4. Which players does it misprice?</h2>
<p>Each point is one player in one window; hover for the name. Up means the next window&rsquo;s
on-court margin liked him <i>more</i> than the rating did. The dark line is the binned average. The
total panel is a U &mdash; wings and bigs are both under-credited and guards over-credited &mdash;
so the mispricing was never a straight line in bigness. The two side panels slope in opposite
directions.</p>
<div class="row"><div id="f4t" class="plot"></div><div id="f4o" class="plot"></div>
<div id="f4d" class="plot"></div></div>

<h2>5. The offense/defense trade, 2024-26, by name</h2>
<p>The top 30 by total rating. For each player the hollow marker is what the box-score-free on-court
signal alone says, and the filled marker is the shipped rating. Both are in standard deviations
above average over the same pool, because the benchmark is a single-window estimate and is shrunk
far harder than the rating &mdash; comparing them raw would show attenuation rather than
disagreement. A long bar is a player the two disagree about.</p>
<div class="row"><div id="f5o" class="plot"></div><div id="f5d" class="plot"></div></div>

<h2>6. Does the boosted correction earn its place?</h2>
<p>How far each archetype sits from where the benchmark puts it, with guards as the reference, before
and after the correction. A bar pointing toward zero is the correction helping. <b>It is a wing
fix, not a big fix</b>: the wing gap on the total closes by a quarter and the offense gaps both
shrink, but the gap on bigs barely moves and on two rows it widens slightly. Whatever is
mispricing big men on each side, this correction is not reaching it.</p>
<div id="f6" class="plot"></div>
<div id="f6txt" class="note"></div>
<button class="tt" data-for="t6">show numbers</button><div id="t6" class="tbl"></div>

<h2>What to be careful about</h2>
<p class="sub">The benchmark is one window of on-court margin, so it is noisy. That pulls every
correlation on this page down by the same amount, which is why the figures compare differences
rather than levels.<br>
A player who stays on the same team shares teammate effects between his window and his benchmark
window. Splitting on whether his main team changed barely moves the answer, so that is not what is
driving any of this.<br>
The bigness index is built from the box score itself: offensive rebounds plus blocks plus a third of
defensive rebounds, minus half of assists and four tenths of threes, all per 36 minutes. Terciles of
it are what &ldquo;guard&rdquo;, &ldquo;wing&rdquo; and &ldquo;big&rdquo; mean here.</p>
</main>
<script>
const D = __DATA__, LIGHT = __LIGHT__, DARK = __DARK__, FONT = __FONT__;
// An ordinal ramp has to be re-stepped for the dark surface, not flipped: the light ramp's darkest
// step sits at about 1.3:1 on #1a1a19 and disappears.  Each set runs low to high contrast against
// its own surface, and every step here clears 3:1.
const STEPS_LIGHT = __STEPS__, STEPS_DARK = ["#3d6fae","#5b9ae8","#a8cdf5"];
const $ = id => document.getElementById(id);
let TH = LIGHT;
const CFG = {displaylogo:false, responsive:true,
             modeBarButtonsToRemove:["select2d","lasso2d","autoScale2d","toggleSpikelines"]};
const SIDEKEY = {"Total":"t","Offense":"o","Defense":"d"};

function base(extra) {
  return Object.assign({
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{family:FONT, size:12, color:TH.text2},
    margin:{l:56,r:16,t:34,b:44}, hovermode:"closest", showlegend:false,
    xaxis:{gridcolor:TH.grid, linecolor:TH.grid, zerolinecolor:TH.zero, zerolinewidth:1,
           tickfont:{color:TH.text2}, title:{font:{color:TH.text2}}},
    yaxis:{gridcolor:TH.grid, linecolor:TH.grid, zerolinecolor:TH.zero, zerolinewidth:1,
           tickfont:{color:TH.text2}, title:{font:{color:TH.text2}}}
  }, extra || {});
}
function tbl(id, head, rows) {
  $(id).innerHTML = "<table><thead><tr>" + head.map(h=>"<th>"+h+"</th>").join("") +
    "</tr></thead><tbody>" + rows.map(r=>"<tr>"+r.map(c=>"<td>"+c+"</td>").join("")+"</tr>").join("") +
    "</tbody></table>";
}

/* 1 - what a rating is made of. Stages of one build-up, so an ordinal ramp, not the side hues. */
function f1() {
  const names = [["prior","Box-score prior"],["boost","Correction"],["resid","On-court residual"]];
  const steps = TH === DARK ? STEPS_DARK : STEPS_LIGHT;
  const traces = names.map(([k,label],i)=>({
    type:"bar", name:label, x:D.build.map(r=>r.side), y:D.build.map(r=>r[k]),
    marker:{color:steps[i], line:{width:2, color:TH.surface}},
    text:D.build.map(r=>r[k].toFixed(2)), textposition:"outside", cliponaxis:false,
    textfont:{color:TH.text2, size:11},
    hovertemplate:"<b>%{x}</b><br>"+label+": %{y:.2f}<extra></extra>"}));
  Plotly.react("f1", traces, base({barmode:"group", bargap:0.35, bargroupgap:0.08, height:330,
    showlegend:true, legend:{orientation:"h", y:1.14, x:0, font:{color:TH.text2}},
    yaxis:{...base().yaxis, title:{text:"spread, points per 100", font:{color:TH.text2}},
           rangemode:"tozero"},
    margin:{l:64,r:16,t:56,b:34}}), CFG);
  tbl("t1", ["Side","Box-score prior","Correction","On-court residual"],
      D.build.map(r=>[r.side, r.prior.toFixed(2), r.boost.toFixed(2), r.resid.toFixed(2)]));
}

/* 2 - the shrinkage sweep. One series, so no legend; the two points that matter are labelled. */
function f2() {
  const k = D.knob;
  const pts = [["shipped","what ships now"],["best","the best this knob can do"]];
  const traces = [{
    type:"scatter", mode:"lines", x:k.lam, y:k.corr,
    line:{color:TH.t, width:2.5},
    hovertemplate:"shrinkage constant %{x:,.0f}<br>agreement %{y:.4f}<extra></extra>"}];
  pts.forEach(([key,label])=>traces.push({
    type:"scatter", mode:"markers+text", x:[k[key].lam], y:[k[key].corr],
    marker:{size:11, color:TH.t, line:{width:2.5, color:TH.surface}},
    text:[" "+label+" ("+k[key].corr.toFixed(4)+")"], textposition:"middle right",
    textfont:{color:TH.text2, size:11.5},
    hovertemplate:label+"<br>%{x:,.0f}<extra></extra>"}));
  Plotly.react("f2", traces, base({
    xaxis:{...base().xaxis, type:"log", tickmode:"array",
           tickvals:[200,500,1000,3000,10000,30000,100000],
           ticktext:["200","500","1,000","3,000","10,000","30,000","100,000"],
           title:{text:"shrinkage constant on the on-court part (log scale)", font:{color:TH.text2}}},
    yaxis:{...base().yaxis, title:{text:"agreement with the benchmark", font:{color:TH.text2}},
           range:[Math.min(...k.corr)-0.004, Math.max(...k.corr)+0.006]},
    height:320}), CFG);
  tbl("t2", ["Shrinkage constant","Agreement with the benchmark"],
      k.lam.map((l,i)=>[Math.round(l).toLocaleString(), k.corr[i].toFixed(4)]));
}

/* 3 - tilt toward bigs. One small panel per side; the hue is the side, the bar is the source. */
function f3() {
  // horizontal bars fill bottom-to-top, so the reference goes last to read at the top
  const labels = ["The box prior","The rating","The on-court evidence"];
  [["f3t","Total"],["f3o","Offense"],["f3d","Defense"]].forEach(([div,side])=>{
    const r = D.tilt.find(x=>x.side===side), c = TH[SIDEKEY[side]];
    const vals = [r.prior, r.rating, r.evidence];
    // hollow means the on-court benchmark, filled means the model - the same convention as figure 5
    const lim = Math.max(...vals.map(Math.abs)) * 1.5 || 1;
    const range = vals.every(v=>v>=0) ? [0, lim] : vals.every(v=>v<=0) ? [-lim, 0] : [-lim, lim];
    Plotly.react(div, [{
      type:"bar", orientation:"h", y:labels, x:vals,
      marker:{color:[c, c, "rgba(0,0,0,0)"], line:{width:[0,0,2], color:c}},
      text:vals.map(v=>(v>0?"+":"")+v.toFixed(3)), textposition:"outside",
      textfont:{color:TH.text2, size:11}, cliponaxis:false,
      hovertemplate:"%{y}<br>tilt %{x:+.3f}<extra></extra>"}],
      base({title:{text:side, font:{color:c, size:14}, x:0, xanchor:"left"},
            height:260, margin:{l:138,r:44,t:36,b:46}, bargap:0.42,
            xaxis:{...base().xaxis, zeroline:true, range:range,
                   title:{text:"tilt toward bigs, per standard deviation", font:{color:TH.text2}}}}), CFG);
  });
  tbl("t3", ["Side","On-court evidence","The rating","The box prior"],
      D.tilt.map(r=>[r.side, r.evidence.toFixed(3)+" ± "+r.evidence_se.toFixed(3),
                     r.rating.toFixed(3)+" ± "+r.rating_se.toFixed(3),
                     r.prior.toFixed(3)+" ± "+r.prior_se.toFixed(3)]));
}

/* 4 - who gets mispriced. Scatter plus a binned average so the shape reads through the cloud. */
function f4() {
  const bins = 9;
  [["f4t","Total","t"],["f4o","Offense","o"],["f4d","Defense","d"]].forEach(([div,side,key])=>{
    const c = TH[SIDEKEY[side]];
    const xs = D.misprice.map(p=>p.b), ys = D.misprice.map(p=>p[key]);
    const lo = Math.min(...xs), hi = Math.max(...xs), w = (hi-lo)/bins;
    const bx=[], by=[];
    for (let i=0;i<bins;i++){
      const a=lo+i*w, b=a+w;
      const sel = ys.filter((_,j)=> xs[j]>=a && (i===bins-1 ? xs[j]<=b : xs[j]<b));
      if (sel.length>4){ bx.push(a+w/2); by.push(sel.reduce((s,v)=>s+v,0)/sel.length); }
    }
    Plotly.react(div, [
      {type:"scattergl", mode:"markers", x:xs, y:ys,
       marker:{size:5, color:c, opacity:0.32, line:{width:0}},
       customdata:D.misprice.map(p=>[p.n,p.w,p.p]),
       hovertemplate:"<b>%{customdata[0]}</b><br>%{customdata[1]} · "+
                     "%{customdata[2]:,} poss<br>bigness %{x:.1f} · gap %{y:+.2f}<extra></extra>"},
      {type:"scatter", mode:"lines", x:bx, y:by, line:{color:c, width:3},
       hovertemplate:"binned average %{y:+.2f}<extra></extra>"}],
      base({title:{text:side, font:{color:c, size:14}, x:0, xanchor:"left"},
            height:300, margin:{l:48,r:14,t:34,b:44},
            xaxis:{...base().xaxis, title:{text:"bigness", font:{color:TH.text2}}},
            yaxis:{...base().yaxis, zeroline:true,
                   title:{text:"benchmark minus rating", font:{color:TH.text2}}}}), CFG);
  });
}

/* 5 - the offense/defense trade. Dumbbell: hollow = on-court only, filled = the shipped rating. */
function f5() {
  [["f5o","Offense","or","oe"],["f5d","Defense","dr","de"]].forEach(([div,side,rk,ek])=>{
    const c = TH[SIDEKEY[side]];
    const rows = D.trade.slice().sort((a,b)=>a[rk]-b[rk]);
    const seg = {type:"scatter", mode:"lines", x:[], y:[],
                 line:{color:TH.grid, width:2}, hoverinfo:"skip"};
    rows.forEach(p=>{ seg.x.push(p[ek], p[rk], null); seg.y.push(p.n, p.n, null); });
    Plotly.react(div, [seg,
      {type:"scatter", mode:"markers", x:rows.map(p=>p[ek]), y:rows.map(p=>p.n),
       marker:{size:9, color:TH.surface, line:{width:2, color:c}},
       hovertemplate:"<b>%{y}</b><br>on-court only: %{x:+.2f} sd<extra></extra>"},
      {type:"scatter", mode:"markers", x:rows.map(p=>p[rk]), y:rows.map(p=>p.n),
       marker:{size:9, color:c, line:{width:1.5, color:TH.surface}},
       customdata:rows.map(p=>[p.p,p.b]),
       hovertemplate:"<b>%{y}</b><br>the rating: %{x:+.2f} sd<br>"+
                     "%{customdata[0]:,} poss · bigness %{customdata[1]:.1f}<extra></extra>"}],
      base({title:{text:side+" — hollow is on-court only, filled is the rating",
                   font:{color:c, size:13}, x:0, xanchor:"left"},
            height:640, margin:{l:150,r:18,t:38,b:44},
            xaxis:{...base().xaxis, zeroline:true,
                   title:{text:"standard deviations above average", font:{color:TH.text2}}},
            yaxis:{...base().yaxis, automargin:true, tickfont:{color:TH.text2, size:11}}}), CFG);
  });
}

/* 6 - does the correction help? Dumbbell from before to after, one row per side and archetype. */
function f6() {
  const rows = [];
  D.correction.forEach(r=>["wing","big"].forEach(a=>rows.push(
    {label:r.side+" · "+a+"s", side:r.side, before:r.before[a], after:r.after[a]})));
  const seg = {type:"scatter", mode:"lines", x:[], y:[], line:{color:TH.grid, width:2},
               hoverinfo:"skip"};
  rows.forEach(p=>{ seg.x.push(p.before, p.after, null); seg.y.push(p.label, p.label, null); });
  const cols = rows.map(p=>TH[SIDEKEY[p.side]]);
  Plotly.react("f6", [seg,
    {type:"scatter", mode:"markers", x:rows.map(p=>p.before), y:rows.map(p=>p.label),
     marker:{size:10, color:TH.surface, line:{width:2, color:cols}},
     hovertemplate:"<b>%{y}</b><br>without the correction: %{x:+.3f}<extra></extra>"},
    {type:"scatter", mode:"markers+text", x:rows.map(p=>p.after), y:rows.map(p=>p.label),
     marker:{size:10, color:cols, line:{width:1.5, color:TH.surface}},
     // put the label on the far side of the "before" marker so the two never collide
     text:rows.map(p=>(p.after<p.before ? (p.after>0?"+":"")+p.after.toFixed(3)+" "
                                        : " "+(p.after>0?"+":"")+p.after.toFixed(3))),
     textposition:rows.map(p=>p.after<p.before ? "middle left" : "middle right"),
     textfont:{color:TH.text2, size:11}, cliponaxis:false,
     hovertemplate:"<b>%{y}</b><br>with the correction: %{x:+.3f}<extra></extra>"}],
    base({height:300, margin:{l:150,r:70,t:20,b:44},
          xaxis:{...base().xaxis, zeroline:true,
                 title:{text:"gap from where the benchmark puts them (guards are the reference)",
                        font:{color:TH.text2}}},
          yaxis:{...base().yaxis, automargin:true}}), CFG);
  $("f6txt").innerHTML = "<p class='sub'>Agreement with the benchmark, without the correction and " +
    "with it: " + D.correction.map(r=>"<b>"+r.side.toLowerCase()+"</b> "+r.before.corr.toFixed(4)+
    " → "+r.after.corr.toFixed(4)+" ("+(r.after.corr-r.before.corr>=0?"+":"")+
    (r.after.corr-r.before.corr).toFixed(4)+")").join("; ") + ".</p>";
  tbl("t6", ["Row","Without the correction","With the correction"],
      rows.map(p=>[p.label, p.before.toFixed(3), p.after.toFixed(3)]));
}

function draw(){ f1(); f2(); f3(); f4(); f5(); f6(); }
function setTheme(mode){
  document.documentElement.dataset.theme = mode==="dark" ? "dark" : "light";
  TH = mode==="dark" ? DARK : LIGHT;
  $("theme").textContent = mode==="dark" ? "light" : "dark";
  draw();
}
$("theme").onclick = () =>
  setTheme(document.documentElement.dataset.theme==="dark" ? "light" : "dark");
document.querySelectorAll(".tt").forEach(b=>b.onclick=()=>{
  const t = $(b.dataset.for);
  t.classList.toggle("on"); b.classList.toggle("on");
  b.textContent = t.classList.contains("on") ? "hide numbers" : "show numbers";
});
setTheme(matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
window.addEventListener("resize", () => draw());
</script>
</body></html>"""


def render(payload: dict, light: dict, dark: dict, steps, font: str, path: Path) -> Path:
    css = _CSS
    # longest key first: "text" is a prefix of "text2", so a naive pass would corrupt "$L_text2"
    for prefix, theme in (("$L_", light), ("$D_", dark)):
        for k in sorted(theme, key=len, reverse=True):
            css = css.replace(prefix + k, theme[k])
    css = css.replace("$FONT", font)
    html = (_HTML
            .replace("__CSS__", css)
            .replace("__PALETTE__", ",".join([light["t"], light["o"], light["d"]]))
            .replace("__NPAIRS__", f"{payload['n_pairs']:,}")
            .replace("__MINPOSS__", f"{payload['min_poss']:,}")
            .replace("__DATA__", json.dumps(payload, ensure_ascii=True))
            .replace("__LIGHT__", json.dumps(light))
            .replace("__DARK__", json.dumps(dark))
            .replace("__STEPS__", json.dumps(list(steps)))
            .replace("__FONT__", json.dumps(font)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
