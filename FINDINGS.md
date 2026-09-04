# Findings: three-year units, the LRBoost prior, and what still tilts the leaderboard

This phase set out to fix player ratings that were unusable: centers swept the top of the 2026
board, and the coefficients were being applied far outside the range they were fit on. Three
changes shipped, two hypotheses were tested and rejected, one design question was settled on
out-of-sample evidence, and one limitation turned out to be real rather than a bug.

## 1. The random effect is now per player per three-season window

It used to be one effect per player-season, so a player got three separate `u` values inside a
window that the ratings then averaged. That is noisier than estimating one three-year effect, and
it did not line up with the playoff block, which was already per window. Box rates moved to the same
unit, with the per-season empirical-Bayes constants blended down by each player's own possessions so
that `k` stays a per-season measurement property and the padding target still tracks in-window
league drift.

Possessions per unit roughly tripled, so every possession-denominated constant was re-tuned.
Cross-validation and REML agreed on both tuning windows:

| constant | was | now |
|---|---|---|
| `lam_beta` | 7113.8 | 10149.5 |
| `lam_ratio_beta` | 0.6124 | 0.5 |
| `lam_plugin` | 23258 | 18351.8 |
| `lam_ratio_plugin` | 0.25 | 0.2872 |

**The coefficients barely moved.** At 2024-26, 3PM went 1.491 to 1.50, 2PM 0.652 to 0.66, 2P-miss
-0.642 to -0.64, FTM 0.720 to 0.71. At 2012-14, 3PM 1.255 to 1.25 and STL 0.768 to 0.78. The
published drift story is unchanged by the unit and lambda change. Mean shrinkage `diag(G(G+lam I)^-1)`
rose from 0.21 to about 0.25, so the on-court residual carries a little more weight than before.

REML also puts the defensive player variance well above the offensive one, `tau2_O` 0.64 against
`tau2_D` 1.93 on 2000-02 and 0.68 against 2.71 on 2012-14. That is why the defensive residual has
about three times the spread of the offensive one. It is what the data says, not a tuning artifact.

## 2. LRBoost: a gradient-boosted correction on the frozen linear prior

The prior enters every stint as a sum over the ten players on the floor, so any per-player function
slots into the mixed model as an offset. Beta is fit first and frozen, so the coefficient figure is
untouched and the booster explains only what beta leaves behind.

Five things had to be right before it did anything useful.

**Pool across windows.** Fit inside a single window the booster is inert: 319 training players per
side, shrinkage 0.26 on offense and 0.03 on defense, a correction with standard deviation 0.06
against a prior standard deviation near 3. Pooled over all ten windows it has about 3000 training
rows per side, and the shrinkage rises to 0.87 and 0.59.

**Weight by the shrinkage diagonal.** The ridge satisfies `Var(u) = tau2 G(G+lam I)^-1` exactly, so
`Var(u_i) = tau2 a_i` and the de-shrunk target `u_i/a_i` has variance `tau2/a_i`. The correct weight
is `a_i`, which is also the Fay-Herriot weight and is collinearity-aware.

**Strip the linear span.** The plug-in fit holds beta fixed while penalising `u`, so the residual
keeps about 8% of the linear prior, consistently across all ten windows. A free booster re-emits it,
silently rescaling coefficients that were frozen on purpose and amplifying the very extrapolation
the stage exists to damp. After residualizing, the correction is curvature only.

**Measure the shrinkage the way you score.** Playing time is in the model to absorb role effects but
is frozen at a starter reference when scoring, so the out-of-fold predictions that `s` is estimated
from must be frozen the same way. Otherwise `s` is credited for an effect that is then thrown away.
Fixing this dropped the offensive permutation null from 0.42 to 0.00.

**Playing time helps one side and hurts the other.** Against five stratified permutation nulls (the
target shuffled within deciles of the weight, so a target keeps its own variance but loses its link
to the features):

| configuration | side | real `s` | null mean | null max | margin |
|---|---|---|---|---|---|
| no playing time | O | **0.868** | 0.069 | 0.343 | **0.800** |
| no playing time | D | 0.572 | 0.097 | 0.334 | 0.475 |
| playing time, frozen | O | 0.646 | 0.037 | 0.183 | 0.609 |
| playing time, frozen | D | **0.589** | 0.015 | 0.073 | **0.574** |

So offense fits without it and defense fits with it, which is what ships. The mechanism is coherent:
defensive residuals are the ones contaminated by opponent quality, since a backup center faces
backups and the ridge under-credits opponents whose own effects are shrunk. That is a
playing-time-correlated nuisance, and offense has much less of it. This is the answer to whether
minutes share belongs in the prior: yes on defense, as a control that never reaches a rating; no on
offense.

**It passes both acceptance tests.** Held-out games, five game-grouped folds, booster fit on the
other nine windows so nothing leaks:

| bucket | prior slope, linear | prior slope, boosted |
|---|---|---|
| under 1500 possessions, defense | 0.841 | **0.942** |
| under 1500 possessions, offense | 0.678 | 0.701 |
| 1500-4500, defense | 0.793 | 0.810 |
| over 4500, defense | 1.016 | 1.032 |
| over 4500, offense | 0.968 | 0.967 |

And on held-out weighted mean squared error, fold-paired: +0.339 ± 0.098 on 2012-14, +0.217 ± 0.158
on 2024-26, pooled **+0.278 ± 0.090, or 3.1 standard errors** in favour of the boosted prior.

## 3. Rejected: diminishing returns on the lineup sum

The design prices a stint by the sum of five players' rates. If a stat is produced collectively and
credited individually, defensive rebounds above all, the team return should diminish and the squared
lineup sum should carry a coefficient opposite in sign to the linear one.

In sample this looks conclusive. In both 2012-14 and 2024-26, every significant squared term opposes
its linear term: 2PM +0.652 against -0.023 (z = -6.2), FTM +0.720 against -0.023, 3PM +1.491 against
-0.079, AST +0.391 against -0.013, defensive DRB and BLK both flipping. Eight of the top ten terms
in 2024-26 are sign-reversed.

**It fails out of sample and it does not fix the board.** Held-out weighted mean squared error is
0.3 to 0.4 basis points *worse* with the 26 squared terms, though better than a shuffled placebo,
which says there is weak real structure that does not pay for the parameters. Applied to the
ratings, the rank correlation with and without is 0.998, and it moves Mitchell Robinson *up*, from
9th to 5th. Dropped.

## 4. Settled: three-season windows, not five

Six five-season windows against the ten three-season ones, over the same thirty seasons, with a
game-grouped five-fold split inside each window so every game is held out exactly once either way.

| | 3-season | 5-season |
|---|---|---|
| held-out weighted MSE, joint fit | **3624.5** | 3624.9 |
| what the box prior buys over a zero-prior RAPM | **30.1 bp** | 24.5 bp |
| mean coefficient standard error | 0.189 | **0.161** |
| drift surviving, mean range over standard error | **3.8** | 3.2 |
| rating stability, adjacent windows | **0.714** | 0.676 |
| prior calibration, under 1500 possessions, defense | 0.679 | **0.728** |

**Three-season wins.** Five years buys 15% tighter standard errors, which is mechanical rather than
evidence of a better model, and a marginally better low-possession calibration. Three years is
better on held-out prediction, resolves more drift relative to its own noise, and is clearly better
on rating stability, 0.714 against 0.676. The plan anticipated a split verdict where five years
suited the coefficients and three the ratings; instead five years blends enough career change into a
single player effect that it loses on both.

The five-year run is still useful as a check on which drift is real. **ORB sharpens** at five years,
range over standard error 5.48 to 6.00, and **BLK sharpens**, 1.85 to 2.66, so that drift is signal.
STL (4.47 to 2.83), FT-miss (4.36 to 2.22), FTM (3.34 to 2.14) and AST (2.38 to 1.19) all shrink, so
that drift was partly resolution. The two headline movers, ORB and 3PM, survive at both lengths.

## 5. What is left, and why it is not a bug

The named cases improved. Mark Williams fell from +5 to +3.98 and out of the top fifty. Paul Reed's
correction went from +0.94 to +0.07, Mitchell Robinson's from +0.68 to -0.06.

But bigs still fill the board and high-usage guards sit lower than they should. Three checks say
this is a limitation rather than a defect:

- **The offense/defense split is identified.** The correlation between a player's offensive and
  defensive residual is +0.09, not the strong negative a seesaw would produce.
- **The prior is calibrated.** Pooled out-of-fold slopes are 0.94 on offense and 0.98 on defense
  against the 0.98 and 0.92 that beta-estimation noise alone predicts. There is no detectable
  over-dispersion to correct.
- **The defensive prior is genuinely defensive.** 82% of its variance comes from blocks, defensive
  rebounds and steals rather than from stats accrued on offense.

The honest reading is that the box score cannot see perimeter defense, so the defensive prior, which
has almost as much spread as the offensive one, is built from the three counting stats that bigs
accumulate. The model then hands centers +4 to +6 on defense and guards -1 to -2, and the on-court
residual is not free enough to overturn it.

**The older eras look right, which localises the problem.** 1997-99 returns David Robinson, Shaquille
O'Neal, Michael Jordan, Grant Hill, Alonzo Mourning, Dikembe Mutombo, Arvydas Sabonis, Karl Malone,
John Stockton and Tim Duncan as the top ten. 2009-11 returns LeBron James, Dwight Howard, Chris Paul,
Kevin Garnett, Dwyane Wade, Yao Ming, Tim Duncan and Manu Ginobili. Nothing is structurally broken;
what stands out in 2024-26 is a modern crop of low-minute, high-efficiency bigs whose per-100 rates
sit far outside the range the coefficients were fit on.

The best evidence that the model knows this is the residual itself. The players the on-court margin
likes most beyond their box score, at 2024-26, are OG Anunoby, Dorian Finney-Smith, Toumani Camara,
Herbert Jones and Marcus Smart: the exact archetype the box score is blind to.

## 6. Open: the empirical-Bayes padding may be too light for the new unit

Rates are shrunk as `(n·rate + k·target) / (n + k)` with `k = sigma2_within / tau2_between` per
feature per season. The `k` values are small in possession terms (assists 63-71, 2PM 65-72, 3PM
114-126, blocks 154-193, steals 574-801), so at 500 possessions a player already keeps 0.75 to 0.88
of his own rate for everything except steals, and at 1500 he keeps 0.90 to 0.96.

It does bite at the very bottom: under 500 possessions the padding cuts the spread of block rate by
52%, offensive rebound rate by 66% and 3PM rate by 47%. But the combined prior is still widest in
that bucket, standard deviation 3.26 against 2.55 for starters, with a 1st percentile of -12.6 and a
maximum absolute value of 19.4. Thirteen features each keeping most of their own sampling noise
compound through the coefficients. The tail is almost entirely negative, so it distorts the bottom
of the board rather than the top, and the leaderboard's 1000-possession floor hides most of it.

Two facts pull in opposite directions and neither has been acted on. `k` is estimated from a
split-half decomposition **within a season** and then possession-blended onto the three-year unit;
between-player variance over three years is smaller than over one, so the correct window-level `k`
is probably larger than what is applied. Against that, heavier padding was tested at checkpoint 4
and made the low-bucket calibration slope worse, which is why it was left light. That test predates
the unit change and is worth repeating.

## 7. The blend was never the problem, and the big-tilt is not a bias

An intermediate diagnosis held that the rating over-weights the box prior by about 1.7x, that
`lam_plugin` was the knob that fixes it, that this was why bigs sweep the 2024-26 board, and that
the LRBoost correction should probably then be deleted as unearned complexity. Tested properly
(`scripts/16`-`20`), all four fail. The criterion throughout is the one that diagnosis proposed:
rank players in window w, score against the same player's window w+1 **pure on-court RAPM**, a
zero-prior fit with no box score anywhere in it, on different games with different teammates.
2133 player pairs with 3000+ possessions in both windows, possession-weighted, demeaned within
window. Beta is held fixed at the same cross-fitted `lam_beta` throughout, so none of this reaches
the coefficient study.

**`lam_plugin` cannot re-blend the rating.** The rating is `prior + u` and the plug-in fit prices
the prior at exactly 1.0 by construction; `lam_plugin` only controls how much `u` is added on top.
Lowering it adds weight to the residual rather than removing weight from the prior, and at the
criterion's optimum the prior still carries 73% of the rating's variance. The 40% figure came from
blending the prior against a *zero-prior RAPM*, which is not the same object as `u`, and no knob in
the pipeline traverses that blend.

**Re-blending is worth +0.007, by any route.** Shipped 0.5739; best in the `prior + u` family
0.5807 at lam 10899, ratio 0.18; best with the prior weight `c` freed as well 0.5809; the ceiling
from regressing the target on (prior, u) with free coefficients 0.5811. The gain is real — bootstrap
over window pairs, 95% CI [+0.004, +0.010] — and inert: Spearman 0.985 between boards, 17/20 top-20
overlap, and the big share of the 2024-26 top 20 rises from 0.70 to 0.75.

**The big-tilt is evidence, not bias.** Put the shipped rating in the regression and add a bigness
term: if the rating over-credits bigs the term is negative. It is **+0.045, z = +4.7**.
Standardising each variable by its own spread so the tilts are comparable:

| tilt toward bigs, per standard deviation | total | offense | defense |
|---|---|---|---|
| next window's pure on-court RAPM (the evidence) | +0.042 | -0.057 | +0.091 |
| the shipped rating | +0.025 | -0.145 | +0.188 |
| the box prior alone | +0.021 | -0.148 | +0.213 |

The total board is tilted toward bigs *less* than thirty seasons of box-score-free on-court margin
independently says it should be.

**What is real is the per-side split, and it cancels in the total.** The rating reproduces the sign
of the on-court evidence on both sides but roughly doubles the gradient. As archetype dummies with
guards as the reference, the rating already in the regression:

| | offense | defense | total |
|---|---|---|---|
| wing | +0.165 (z 4.3) | -0.033 (z -0.6) | +0.299 (z 4.3) |
| big | +0.226 (z 5.5) | -0.280 (z -4.2) | +0.289 (z 4.0) |

The offensive rating under-credits bigs, the defensive rating over-credits them by almost exactly as
much. Section 5's mechanism was right: the box score cannot see perimeter defense, so the defensive
prior is built from blocks, defensive rebounds and steals. The site shows the two sides separately,
so this is a real defect even though the headline board is sound. Note also that on the total, wings
and bigs are *equally* under-credited (+0.299 and +0.289) — the mispricing is U-shaped in archetype,
so the one-dimensional bigness framing was never the right one, and the group the total board
over-credits is guards.

**The booster is the most valuable component, not the one to cut.** It is worth **+0.0129** on this
criterion, 1.9x the entire re-blending question, and it is the only piece that moves the real bias:
its mean correction is +0.114 offense for wings against -0.089 and -0.097 for guards and bigs, the U
shape learned rather than imposed. Once it is in, re-tuning `lam_plugin` is worth +0.003 and the
optimum moves *up* from 18573, not down.

**But it is a wing fix, not a big fix.** With the correction in, the archetype dummies go: total
wing +0.298 to +0.229, offense wing +0.165 to +0.114, offense big +0.226 to +0.207, defense big
-0.279 to -0.269 — and total big +0.288 to +0.298, defense wing -0.033 to -0.055, both slightly
*worse*. It closes about a quarter of the wing gap and barely touches bigs. Whatever misprices big
men on each side, this correction does not reach it. An earlier version of this section said it
"shrinks all three and eliminates none", which overstated what it does for bigs.

*(These figures were re-measured on the corrected convention. The first pass reported +0.0174 and
2.6x; it added the correction on top of a residual fit without it, which is the double-count
described in section 8.)*

**So no constant changed.** `lam_plugin` 18351.8 and ratio 0.2872 sit on the plateau of the
player-level criterion (0.5912 against a 0.5941 argmax that is flat from 18573 to 53940).

Caveats. The target is a single-window estimate and therefore noisy, which attenuates every
correlation equally; the conclusions are about differences, not levels. A player's `u` in window w
shares teammate contamination with his w+1 target if he stays put, but splitting on whether the
dominant team changed barely moves the optimum (8349 for movers, 10899 for stayers), so that channel
is not driving the result. The booster is cross-fitted on player id so a player never scores himself,
though his former teammates can still be in its training set.

## 8. A double-count in the ratings table, and three smaller fixes

`scripts/10_boost.py` refits the on-court residual with the correction carried as an offset, which
is the coherent thing to do: the residual should be what the margin says beyond both the box score
*and* the correction. `scripts/08_ratings.py` did not. It took a residual fit without the offset and
added the correction on top, so it counted whatever the correction already explained twice.

The two files were therefore not the same table:

| | max absolute difference |
|---|---|
| `prior_total`, `boost_total` | 0 |
| `u_total`, `rating_total` | 1.385 |
| `rating_off` / `rating_def` | 1.117 / 1.210 |

The site read `ratings_boosted.parquet`, so the published board was the correct one; the
downloadable `player_ratings.csv` was the wrong one. Overall Spearman 0.9962 and 2024-26 top-20
overlap 18/20, so it never showed up as an obviously broken leaderboard, but individual players
moved a long way — Trae Young 408 to 347, Andre Drummond 186 to 126.

`player_ratings_table` now takes the correction as `prior_offset` and threads it into the plug-in
fit, and reports both residuals: `u_*` (fit with the correction, what the rating uses) and
`u_plain_*` (fit without it, so `rapm_mm_*` still means "what the model says with no correction at
all"). `08_ratings.py` is the one canonical builder and `07_plots.py` reads its output. The two
tables now agree to 3e-13.

**This changed a published conclusion.** Section 7's value for the booster was measured on the
double-counting convention. Re-measured properly it is +0.0129, not +0.0174, and the correction
turns out to help wings and barely touch bigs.

Three smaller fixes alongside it. `chimeraboost` was imported by `boost.py` but missing from
`pyproject.toml`, so a fresh clone could not run the booster at all. `site.py` substituted its theme
tokens by naive string replacement in dictionary order, so `$L_text` matched inside `$L_text2` and
the page emitted `--text2:#0b0b0b2`, an invalid colour that silently dropped every caption, footer
and axis label to an inherited one; keys are now replaced longest-first. And `plots._page` and
`plots.index_page`, both unreachable since the two pages were merged into one, are gone.

## 9. The right loss for ranking players is not the loss we used

`c_def` scales the defensive box prior in the plug-in offset. Swept on 2024-26 against two losses,
same folds, same everything else:

| `c_def` | held-out stint MSE | vs shipped (fold-paired) | z | rank vs consensus | defensive rank |
|---|---|---|---|---|---|
| 1.0 (ships) | 3993.85 | 0 | — | 0.779 | 0.750 |
| 0.5 | 3994.32 | +0.47 | 1.1 | 0.846 | 0.830 |
| 0.1 | 3995.90 | +2.05 | 2.9 | 0.887 | 0.885 |
| 0.0 | 3996.46 | +2.61 | **3.4** | **0.892** | **0.887** |

Held-out stint MSE **prefers the shipped setting, at 3.4 standard errors**, over the setting that
raises player rank agreement from 0.78 to 0.89. It is not merely uninformative about this parameter;
it is confidently wrong about it. And the whole sweep moves it by 2.6 parts in 3994 — 0.065%.

The reason is structural. A stint row observes only the **lineup sum** of player effects, so the
stint likelihood decomposes into two parts:

- the **row space** of the design, where individual effects are identified by teammates varying
  across lineups. Held-out stint MSE is a proper and efficient loss here.
- the **near-null space**: reallocating credit between players who share the floor barely changes
  any lineup sum they both appear in. Stint MSE is nearly flat along it.

Ranking players is a loss over the player vector itself, not over its lineup sums, so it needs
information about the null space. `c_def` moves almost purely in the null space, which is why the
loss cannot select it — and why the tiny row-space component it does see (the defensive prior
genuinely does carry some lineup-level signal) makes the loss point the wrong way with confidence.

**This is also why the coefficient study is sound and the ratings were not.** Beta is a lineup-level
estimand validated with a lineup-level loss: perfectly matched. The ratings are a player-level
estimand that was validated with the same lineup-level loss: mismatched. One loss, two different
estimands.

### The general rule

For every tunable, ask which subspace it moves. If it moves the design's null space, no in-sample
loss can select it, and you must either measure it externally or admit you are imposing a belief.

`c_def = 1.0` was never estimated. It was a belief — "the box score measures defense as reliably as
it measures offense" — imposed silently by the plug-in construction. The data never had a chance to
disagree.

This also splits the two guiding principles cleanly. *Adjust for sample size and luck as smartly as
you can* applies in the identified directions, where REML and cross-validation genuinely work. In
the unidentified directions there is no amount of data that helps; there is only a prior, and the
honest move is to state it and source it from outside.

### What to use instead, in order of how much attribution information it carries

1. **Roster-change outcomes.** Predict a team's change in point differential from the ratings of the
   players in and out. The lineup context is genuinely new, so misattribution cannot hide, and it is
   the decision a rating actually supports. Low sample, unbiased, and not yet built here.
2. **Hold out by lineup structure, not by game.** Game-grouped folds preserve every five-man unit
   exactly, so they never test attribution. Holding out whole units, or up-weighting rare lineup
   combinations, puts cost on misattribution using data we already have. Also not yet built.
3. **Next-window on-court, restricted to players who changed teams.** Teammate churn decorrelates the
   target's attribution error from the source's. The unrestricted version is what section 7 used, and
   its lack of power is exactly this: most players stay put, so most pairs carry the error in both.
4. **An external consensus.** It has player-level information our data does not (tracking, matchups).
   Not ground truth — it is other people's models — but the only independent attribution signal
   available at scale.
5. **Held-out stint MSE.** Correct for beta. For anything player-level, a guard against blowing up,
   never a selector.

## 10. Wins Produced in, xRAPM out: what re-pricing the prior actually buys

The framing that cracked this: the board fails in the **Wins Produced** direction (rebounders float
to the top) and needs to succeed in the **xRAPM** direction.

The mechanism is exact. `beta` is estimated by regressing stint margin on the **lineup sum** of box
rates. That answers a team-economics question — what is a team's rebounding worth — and the answer
is correct. We then apply it to individuals, which assumes the man who collected the rebound is the
man who created it. That is precisely the Wins Produced move. xRAPM never asks what a rebound is
worth; it asks what a player's box line predicts about **his own RAPM**, which discounts conserved
stats automatically.

So we fit the second thing: the same 13 rates per side regressed on a pure on-court RAPM at the
player level, de-shrunk with the Fay-Herriot weight, leave-one-window-out (`scripts/27_xrapm_prior.py`).

**How much player-level signal is there at all?** Weighted R-squared of a player's own 13 rates on
his own on-court RAPM, 2000+ possessions:

| offense | defense |
|---|---|
| **0.533** | **0.258** |

**The result splits by side, and that is the finding.** Rank agreement with the consensus on 2024-26:

| prior | offense | defense | total |
|---|---|---|---|
| team-level (ships) | **0.869** | 0.750 | 0.779 |
| player-level (xRAPM style) | 0.836 | 0.804 | 0.802 |
| player-level, defensive weight 0 | 0.836 | 0.882 | 0.852 |
| **team-level offense, no defensive prior** | **0.868** | **0.886** | **0.890** |

Re-pricing helps defense a lot (0.750 to 0.804, and to 0.882 once its weight is also cut) and
**hurts offense** (0.869 to 0.836). The reason is the conservation table in section 9: shooting and
turnovers survive into the lineup sum at 0.83 to 0.90, so for offense the lineup-level price *is*
the player-level price, and the lineup regression estimates it from every stint row rather than from
a noisy shrunk RAPM target. Rebounds and blocks survive at 0.59 to 0.69, so for defense the transfer
fails outright.

**The rule, stated once:** a lineup-level coefficient transfers to individuals exactly to the extent
the stat is not conserved. Our offensive prior is already an xRAPM-quality prior because offensive
box stats are individually attributable. Our defensive prior is a Wins Produced prior because
defensive box stats are collective outcomes credited to whoever collected them.

**What this means for the fix.** Do not re-fit all 26 coefficients against RAPM — that trades a good
offensive prior for a worse one. Keep the published team-level beta for offense, and drop the
defensive prior to near zero. Even correctly re-priced, the defensive box score tops out around 0.26
R-squared, and adding it at any weight injects more archetype bias than it repays.

The honest way to say it: **without tracking data the box score has no defensive vocabulary, so our
defensive rating should be close to pure RAPM.** That is what the xRAPM family effectively did on
defense before tracking existed too.

**Caveat specific to defense.** The consensus blends metrics whose defensive components are
themselves more RAPM-driven than their offensive ones, so "pure RAPM scores best on defense" carries
some circularity risk. The offensive control argues against a general circularity — there the box
prior beats the on-court residual 0.83 to 0.37 — but it does not fully rule it out on defense alone.
A roster-change test (section 9, item 1) would settle it and has not been built.

## 11. The architecture is wrong in one specific way: it adds where it should blend

Three corrections came out of pushing on the offensive side.

**First, section 10 was unfair to the player-level prior.** That fit used the same window's RAPM
de-shrunk by 1/a, effectively unpenalized, on 13 correlated rates - a high-variance estimate. Redone
with a leak-free target (a player's rates in window w against his own on-court impact in w+1) and a
cross-validated ridge (`scripts/28_offense_prior.py`), the two ways of pricing the box score are a
dead heat, not a win for either:

| box prior alone, vs consensus | offense | defense |
|---|---|---|
| team-level (lineup regression) | 0.829 | 0.506 |
| player-level, trained on next-window impact | 0.825 - 0.827 | 0.503 - 0.519 |

So "re-price it the xRAPM way" is not the lever. Both priors are equally good, and equally limited.

**Second, diminishing returns on usage does not show up at the player level.** The project rejected
squared lineup-sum terms earlier on held-out stint MSE; section 9 voids that rejection, so the
question was re-opened against a player-level target. Adding usage and usage squared moves
out-of-fold rank agreement by +0.001 on both sides, and the squared usage coefficient comes out
**positive** on offense (+0.141) - increasing returns, not diminishing. Squared scoring rates add
+0.004 on offense and +0.012 on defense. Note this tests a main effect only. The claim that scoring
is worth less *in particular lineups* is an interaction with teammate composition, and that is still
untested.

**Third, and this is the real one.** Each source scored alone against the consensus on 2024-26:

| | box prior alone | pure on-court RAPM alone | what we ship (prior + residual) |
|---|---|---|---|
| offense | 0.829 | 0.765 | **0.851** |
| defense | 0.506 | **0.877** | 0.755 |

On offense, combining beats either piece: the architecture works. **On defense, our shipped rating
is worse than pure RAPM would be on its own.** Adding the box prior does not dilute the on-court
signal, it destroys it - 0.877 down to 0.755.

The cause is that `rating = prior + residual` **adds** two sources. That is only the right
combination when the prior is unbiased and the shrinkage already encodes its variance. Ridge
shrinkage handles noise; it does not handle bias, and the defensive prior is biased along archetype.
Fitting the optimal blend instead, leave-one-team-out so no player is scored by weights his own team
helped set:

| | weight on box prior | weight on pure RAPM | blended |
|---|---|---|---|
| offense | **1.157** | 0.778 | 0.880 |
| defense | **0.116** | 0.908 | 0.877 |

Total rank agreement **0.784 to 0.884**, and archetype bias +0.633 to +0.167. The weights are not
tuned by hand: they are what reliability-weighting the two sources produces, and they reproduce the
per-side answer (offense box-heavy, defense almost pure RAPM) with no fudge factor.

**So the architecture is right in shape and wrong in one operator.** Prior-informed RAPM is the
correct frame. What it needs is a per-side, reliability-weighted blend of the box prediction and the
on-court estimate, with weights measured out of sample against a player-level target - not a sum,
and not a shrinkage constant chosen on possession prediction.

## 12. The determination: our machinery is worse than the textbook

The consensus CSV is validation only from here. Nothing in this section is tuned on it, selected by
it, or fit to it. Every penalty and offense/defense ratio below was chosen by an internal criterion
- how well a candidate predicts a player's own next-window pure on-court impact, pooled over the
nine window pairs that end before 2024-26. The consensus is read once, at the end.

The earlier "reliability blend" at 0.884 is withdrawn as a candidate. Leave-one-team-out or not, its
weights were fit against the validation target, so it is an upper bound on what reweighting could
buy and not a proposal.

Named for what each thing is (`scripts/30_ladder.py`), scored by Spearman rank correlation on the
same 475 players:

| candidate | total | offense | defense |
|---|---|---|---|
| 1. RAPM, no prior | 0.757 | 0.795 | **0.859** |
| 2. box score, team-priced, no on-court term | 0.669 | 0.829 | 0.506 |
| 3. box score, player-priced, no on-court term | 0.707 | 0.828 | 0.500 |
| 4. prior-informed RAPM, team-priced prior | 0.786 | 0.868 | 0.785 |
| **5. prior-informed RAPM, player-priced prior** | **0.844** | **0.884** | 0.846 |
| 6. prior-informed RAPM + boosted correction (ships today) | 0.787 | 0.852 | 0.772 |
| 7. hybrid: player-priced prior on offense, no prior on defense | **0.853** | **0.884** | **0.859** |

**We cannot beat plain prior-informed RAPM. Candidate 5 is textbook prior-informed RAPM and it
scores 0.844; what the project ships scores 0.787.** Everything the project added on top of the
plain construction is net negative. That determination is the point of this section.

**The booster does not earn its place.** At identical penalty and ratio, so the only difference is
whether the correction rides in the offset:

| | total | offense | defense |
|---|---|---|---|
| without the correction | 0.786 | **0.872** | 0.768 |
| with the correction | 0.787 | 0.852 | 0.772 |

A wash on the total and clearly worse on offense. Section 7 measured it at +0.0129 against the
next-window criterion; against an external benchmark it is worth nothing. That is a second instance
of the same lesson - the internal criterion is not neutral.

**The one real improvement is not machinery, it is how the prior is built.** Team-priced to
player-priced moves the total from 0.786 to 0.844. Note carefully that as *standalone metrics* the
two priors are indistinguishable (candidate 2 against candidate 3: 0.829 and 0.828 on offense, 0.506
and 0.500 on defense). The difference only appears when the prior is used as a **shrinkage target**,
because there what matters is not overall rank quality but being unbiased in the directions the
on-court data cannot resolve. The team-priced prior is biased exactly there; the player-priced one
is much less so. This is the standard xRAPM construction, not an invention of ours.

**And the box score should not touch defense.** Candidate 1 beats candidate 5 on defense, 0.859 to
0.846, so even a correctly player-priced defensive prior is a net negative. Candidate 7 simply uses
the prior where it helps and drops it where it does not.

Honesty about candidate 7: the decision to drop the defensive prior is one bit of information taken
from the validation set. It is a structural choice rather than a fitted parameter, and section 10
gives an independent mechanism for it, but it is not zero.

Archetype bias remains in all of them: +0.62 as shipped, +0.49 for candidate 5, +0.35 for candidate
7. Nothing here fully fixes the tilt; it roughly halves it.

## 13. Shipping the hybrid, and the criterion that could not choose the defensive shrinkage

Section 12 recommended the hybrid — the box prior priced to predict a **player** on offense, and no
box prior at all on defense — at total 0.853. That number was read off `outputs/ladder.parquet` by
hand; row 7 never existed in code. Writing it down changed both the construction and the result.

### The construction is one fit, not two

The ladder built each candidate as a separate plug-in fit, so a hybrid looked like two fits with one
side taken from each. It is not. The box term enters the model as an offset `Xbox @ beta`, and the
offensive and defensive box columns are separate, so **zeroing beta's defensive half is exactly "no
defensive prior"** — one fit, one penalty, no per-side machinery. `windows.hybrid_beta` returns that
vector; `scripts/08_ratings.py` passes it to the existing `player_ratings_table` unchanged.

The single fit and the spliced version are not identical (Spearman 0.980 on defense, max difference
0.58 points) because the offensive offset shifts the shared fixed-effect block, and the single fit
is the better of the two. But the reason to prefer it is that it is the construction, not a splice.

### The internal criterion cannot select the defensive shrinkage

The defensive columns are penalised by `lam * lam_ratio` — `estimator._scale` divides them by
`sqrt(lam_ratio)`, so a scalar ridge on the scaled design is that product on the raw one. Sweeping
that single quantity with the offense held fixed (`scripts/33_hybrid.py`,
`outputs/csv/defensive_penalty_path.csv`):

| effective defensive penalty | internal criterion | consensus | defensive spread |
|---|---|---|---|
| 862 | 0.4799 | 0.837 | 2.24 |
| 1500 | **0.4813** | 0.857 | 1.95 |
| 1723 ← *ladder row 7* | 0.4797 | 0.859 | 1.86 |
| 3000 | 0.4802 | 0.877 | 1.54 |
| **5271 ← `lam_plugin * lam_ratio_plugin`, what ships** | 0.4698 | 0.877 | **1.19** |
| 6000 | 0.4714 | **0.886** | 1.13 |
| 18352 | 0.4380 | 0.868 | 0.61 |
| 50000 | 0.3990 | 0.824 | 0.31 |

The internal criterion — rank agreement with the player's own next-window pure on-court impact — is
**flat across the four weakest penalties** (0.4789 to 0.4813, well inside noise) and then declines
monotonically. It has no interior optimum. It cannot pick a value; it picks whichever weak-end grid
point wins a coin flip, and that is where ladder row 7's knobs came from.

This is the same failure as section 9. There, held-out stint MSE preferred the wrong end of the
defensive prior weight at 3.4 standard errors while moving 0.065%. Here the next-window on-court
benchmark is built from the same margin data as the estimate and shares its blind spot, so weaker
shrinkage always looks at least as good against it. **Two of the three internal criteria this
project has tried are blind to the defensive shrinkage, in the same direction, for the same reason.**

### What we shipped, and why it is not hindsight

We did not take the consensus argmax at 6000 — that would be fitting to the validation set. We kept
`lam_plugin * lam_ratio_plugin` = 5271, the constant the project already used, chosen in an earlier
session by `scripts/03_cv.py` and `scripts/16_tune_plugin.py` before any of this existed. Not moving
it is the only option available here that is not chosen with knowledge of the answer.

That has to be stated plainly: **by the time the penalty path was measured, the consensus column was
on the screen.** Any defensive penalty selected now would be contaminated. Keeping the status quo is
defensible precisely because it was not selected now. The +0.04 sitting at 6000 is real and
unexploited, and taking it needs a new internal criterion designed without reference to the
consensus — which is a genuinely open problem, given that the two obvious criteria both fail.

### Result

Against the consensus, 2024-26, 475 players with 1000+ possessions, read once:

| | ships today | **hybrid** | change |
|---|---|---|---|
| total | 0.784 | **0.896** | **+0.112** |
| offense | 0.851 | 0.879 | +0.028 |
| defense | 0.755 | **0.888** | **+0.133** |
| offensive spread | 1.05 | 0.99 | — |
| defensive spread | 2.13 | **1.23** | −0.90 |
| archetype bias | +0.63 | **+0.20** | −0.43 |

Section 12 predicted +0.067; the delivered figure is +0.112, and the difference is entirely the
defensive penalty — 0.853 at ladder row 7's knobs, 0.896 at the shipped ones.

**Five of the six strict xfails in `tests/test_vs_consensus.py` flipped** and are guards now:
defensive spread, defensive agreement, archetype bias, star guards, overall agreement. Stephen Curry
went from buried to 15th, Devin Booker to 21st, LaMelo Ball to 42nd.

One remains: pure on-court defensive RAPM still rates backup bigs above the consensus — Robert
Williams 68th against 134th, Jonathan Isaac 55th against 156th, Luke Kornet 33rd against 76th.
Removing the box prior halved the tilt; it did not remove it. That residue is an **attribution**
question — how credit for a defensive possession is split among five players — and no choice of
prior was ever going to answer it. It is the reason the next piece of work is the luck-adjusted
target rather than another prior.

### Also changed

The boosted correction is out of the shipped rating (`ratings_prior.boost: false`); the ladder
measured it at 0 on total and −0.020 on offense. `boost_*` stays in `player_ratings.parquet` as a
comparison column, and the offset machinery stays, because the correction must be carried into the
fit rather than added afterwards whenever it is used at all (section 8).

The playoff variant applies the pooled delta to the **offensive half only**. The delta was fit as a
change in the team-priced coefficients; adding all of it would make the defensive half non-zero and
quietly reintroduce the prior the hybrid exists to remove.

## 14. Stage 0 of the luck work: HANDOFF's rebound anchors are the wrong convention

Before writing any per-possession counter, `scripts/35_attempt_defs.py` measured what an "attempt"
has to mean for the geometric series `V = x / (1 - m*r)` to be well posed. It is not a free choice,
and the numbers HANDOFF quotes are not the ones the model needs.

**The test.** Every possession ends in an attempt, a turnover, or nothing, and a possession that
reaches an attempt has exactly one more attempt than it has offensive-rebound continuations. So

    (attempts - continuations) + turnovers + leftover = possessions

and each candidate definition predicts a `leftover` it cannot account for. Possessions come from the
already-built stints, so this is an external check, not a self-consistency one. (The `1/(1-m*r)`
form of the multiplier is the same statement rearranged and agrees identically for every definition,
so it tests nothing — an earlier version of this script used it and found all definitions perfect.)

60 games each of 2024, 2005 and 1998:

| definition | attempts/game | OREB% | multiplier | unexplained poss/game |
|---|---|---|---|---|
| A  FGA only, player OREB — **the box-score convention** | 165.6 | 27.2 | 1.177 | **+16.9 (9.0%)** |
| B  FGA only, incl. team OREB | 165.6 | 34.3 | 1.234 | **+23.4 (12.4%)** |
| C  + shooting-foul free-throw trips, and-1 excluded | 181.6 | 32.5 | 1.209 | +7.4 (3.9%) |
| D  + non-shooting trips | 188.5 | 32.0 | 1.199 | **+0.49 (0.25%)** |
| E  D with period-end team OREB dropped | 188.5 | 31.0 | 1.189 | **−0.91 (0.49%)** |

**A and B fail, and A is the one HANDOFF quotes.** A shot that draws a shooting foul records no FGA,
so counting only FGA misses those possessions entirely — 9% of the total. HANDOFF's "OREB% 24.4,
multiplier about 1.15" is definition A on a modern season (2024 gives 23.0 and 1.137). It is a
correct box-score number and the wrong input for this model.

**The anchors are era-dependent**, so no single value can gate the build:

| | 2024 | 2005 | 1998 |
|---|---|---|---|
| OREB% (definition E) | 27.7 | 31.6 | 33.8 |
| multiplier | 1.158 | 1.195 | 1.213 |

Offensive rebounding fell six points over 26 years. Any stage-1 test has to be per-season.

**Two smaller corrections.** The claim that only 89.4% of missed field goals are followed by a
rebound is a `shift(-1)` artifact: a forward scan of at most seven events resolves 5395 of 5396
misses in 1998 and every one of them in 2024, with 95-96% at the very next event. Genuinely terminal
misses are effectively zero. The correction that does matter is the opposite one — **offensive TEAM
rebounds as the period expires, about 5% of all continuations**, which are not continuations at all
because no further attempt followed.

**D versus E is not resolved by this test** (0.25% against 0.49%, and the sign of the residual flips
by era). We take E, because a period-end team rebound is not a continuation on the merits.

**Correction, from section 15.** The reconciliation above is biased, and the bias was invisible
without possession boundaries. `attempts - continuations + turnovers = possessions` double-counts any
possession that took an attempt **and then** turned the ball over -- rebound your own miss, then lose
it -- which is 0.91% of possessions, about 1.8 a game. Every leftover in the table is therefore
understated by roughly that much. The ranking is unaffected, since the gap between D/E and A/B is 9
to 12 percentage points, but definition E's apparent 0.004% fit on 2024 was luck rather than
precision. The stage 1 counters measure that overlap directly instead of absorbing it, and their
values supersede the OREB% and multiplier estimates here.

## 15. Stages 1 and 2 of the luck work: the counters reconcile, and the lineup rates are real

### Stage 1: per-possession counters

`src/eracoef/stints.py` now carries 33 counters per possession, summed per side into the stints
(`POSS_COUNTERS`). Two identities hold and are tested:

    points - pts_tech == 2*(fgm - fg3m) + 3*fg3m + ftm
    reb_cont - cont_dead + att_retained == att - 1        (whenever a possession reaches an attempt)

The second one is where the work was. It says every attempt after a possession's first has to be
accounted for, and getting it to hold flushed out four things the spec did not anticipate. Each was
found by the identity failing, not by reading code:

1. **A shooting foul on a shot the feed ALSO recorded as a missed FG.** Usually a foul on a miss
   records no FGA at all — that is the whole reason free-throw trips count as attempts — but when
   both rows appear they are *one* attempt, and the whistle means that "miss" was never a live
   rebound either. 141 a season.
2. **Possession-retaining fouls.** A flagrant on a made shot hands the ball back, and the next
   attempt follows no rebound at all. That is a third way to get an extra attempt, so the identity
   needs the `att_retained` term rather than a fudge.
3. **Continuations that lead nowhere** (`cont_dead`): rebound your own miss and then turn it over, or
   have the period expire. 0.9% of possessions.
4. **Misses with no rebound row** (`oreb_unattr`): a block or a tip where play plainly continued.

A residue of about 30 possessions a season (0.01% of attempts) still comes out ±1, in exotic foul
sequences. That is bounded — each can move one possession's expected points by at most one attempt —
so `scripts/36_counter_check.py` carries it as a declared 0.05% tolerance rather than pretending the
identity is exact.

**Against the box score**, which is independent data and therefore the check that matters: FGA,
FG3M, FTA and FTM all land at 0.9993-0.9996 of their box totals, a spread of 0.0002. (The level is
below 1 because stints drop invalid possessions.) Possession-level turnovers exceed box turnovers by
1.39 a game, which is team turnovers — shot clock, five-second — exactly as it should be.

**A schema version** (`STINT_SCHEMA`) now lives in the diag frame and is checked on load. The cache
was keyed on path existence alone, so adding a column left every stale file loading happily with the
new columns silently absent, then turning into NaN on the first `concat` and flowing straight into
`y` and `w`. It now raises `StaleStintCache` with the rebuild command.

Measured, per season, on the definition chosen in section 14:

| | 2024 | 2025 | 2026 |
|---|---|---|---|
| possessions reaching an attempt | 87.2% | 86.7% | 86.5% |
| OREB% | 27.13 | 28.10 | 29.07 |
| continuation multiplier | 1.1355 | 1.1413 | 1.1471 |
| first attempt is a three | 37.0% | 39.4% | 38.5% |

### Stage 2: the four-factor RAPMs, and the go/no-go

`src/eracoef/factors.py` fits eFG%, TOV%, OREB% and FT rate as RAPM targets. The target is now a
parameter of `build_design` (`TARGETS` in `design.py`), which is a one-line change at the single
place `y` and `w` were ever defined — but **not** the "only the target column changes" HANDOFF
promised: the denominator is also the row weight and the row filter, because an eFG% row carries
information in proportion to its attempts, not its possessions.

**The gate was split-half reliability of the lineup sum**: fit on half A, fit on half B, correlate
the two rates over the same rows. If a lineup's expected rate does not replicate across interleaved
halves of the same season it is noise, and stage 4 is dead. It is not noise:

| factor | lam | lam_ratio | calibration | sd across lineups | **split-half r** |
|---|---|---|---|---|---|
| eFG% | 3495 | 1.50 | −0.03% | 2.23 | **0.666** |
| TOV% | 2176 | 0.75 | −0.05% | 2.20 | **0.754** |
| OREB% | 414 | 3.00 | −0.55% | 4.29 | **0.738** |
| FT rate | 1355 | 1.00 | −0.10% | 4.21 | **0.762** |

Against a threshold of 0.30. Calibration is within 0.55% everywhere, no rate needed clipping, and
half-fit spread is 0.89-0.94 of full-fit spread, so two-fold cross-fitting compresses the rates only
slightly — which understates the luck adjustment rather than biasing it.

**The offense/defense asymmetry came out estimated rather than imposed**, which is what HANDOFF
asked for and the one result here that is interesting in its own right. `lam_ratio` is
`lambda_D / lambda_O`, so above 1 means the defensive effects are shrunk harder — there is less
defensive skill to find:

- **eFG% 1.50** — defences have relatively little shot-making suppression skill. This is the
  asymmetry an earlier draft proposed hard-coding. It did not need to be hard-coded.
- **OREB% 3.00** — defensive rebounding is much less of an individual skill than offensive
  rebounding, at lineup level.
- **TOV% 0.75** — the one factor where defence dominates. Forcing turnovers is a real defensive
  skill, more so than the offense's ability to avoid them.
- **FT rate 1.00** — symmetric.

**One methodological catch worth recording.** OREB% first selected `lam_ratio = 2.0`, the top of
`config.yaml`'s grid. A criterion sitting on its boundary has not chosen anything — the exact
failure that produced ladder row 7's defensive knobs in section 13. Widening the grid moved it to
3.0, interior. `factors.py` now uses its own wider ratio grid and flags any selection that lands on
an edge. **The lesson from section 13 generalised within one session: always check whether the
argmax is interior before believing it.**

## 16. The defensive box prior improves prediction and worsens attribution, and both are real

The project owner proposed the criterion this section rests on: fit ratings on some seasons, predict
the stints of a season they never saw, score by possession-weighted squared error in points per 100.
It is implemented in `scripts/38_yoy.py` as leave-one-season-out with a symmetric training block --
hold out H, train on {H-1, H+1} for K=2 or {H-2..H+2} minus H for K=4 -- so the held-out season is
identical across every method and every K, and aging cancels because the block brackets H.

**This is the first criterion here that is both external to the model and legal to select on.** It
scores against actual points, so unlike the next-window on-court benchmark it does not share the
estimate's blind spot; and it uses no outside data, so unlike the consensus CSV it can be optimised
against without circularity.

### It ranks the shipped hybrid below what the hybrid replaced

Pooled over 28 held-out seasons, paired by season:

| contrast | stint MSE | z | seasons won |
|---|---|---|---|
| hybrid vs PI-RAPM (team-priced) | **+1.19** | +4.88 | **5 of 28** |
| hybrid + xPTS(ft) vs hybrid | −0.15 | −3.51 | 21 of 28 |
| PI-RAPM vs RAPM with no prior | −8.54 | −14.1 | 28 of 28 |

Every prior beats no prior by a wide margin. But the hybrid -- which the consensus rates 0.896
against PI-RAPM's 0.784 -- loses here, consistently. It is not a calibration artifact: after fitting
optimal per-side scalars PI-RAPM still wins, and the hybrid is the better-calibrated system of the
two (0.93/0.96 against 0.86/0.80). At team-game aggregation the order flips to the hybrid, but not
significantly (z = -1.34).

Two explanations were tested and one survived.

**Not a coverage problem** (`scripts/39_why.py`). The hybrid gives a lightly-used player almost
nothing on defense, so the deficit ought to sit with fringe players. It does not: the hybrid is worse
in every exposure bin and *worst* among established ones (+1.63, z = +5.07, winning 4 of 28 seasons
at 1500-3999 training possessions), against +0.97 and not significant for players with no training
exposure at all.

**Partly a credit-transfer problem** (`scripts/40_movers.py`). Splitting held-out rows by how many of
the ten on the floor changed team since the training block:

| movers on the floor | hybrid - PI | relative to that group's MSE | share of possessions |
|---|---|---|---|
| none | +1.94 | 5.9e-4 | 5% |
| 1-2 | +1.38 | 3.9e-4 | 40% |
| 3+ | +0.91 | 2.4e-4 | 54% |

The gap halves as rosters turn over, which is the signature of PI-RAPM holding credit that does not
travel with the player. But it does not reverse, so PI-RAPM's advantage is not purely artifact.

### Sweeping the weight: the criterion has an interior optimum

`scripts/41_defweight.py` puts one scalar `c_def` on the defensive half of the team-priced beta,
holding the player-priced offensive prior fixed. `c_def = 0` is the shipped hybrid, `c_def = 1` is
prior-informed RAPM.

| c_def | held-out MSE | paired vs c_def=0 | consensus total | consensus defense | archetype bias |
|---|---|---|---|---|---|
| 0.00 | 3627.97 | — | **0.896** | **0.888** | **+0.195** |
| 0.25 | 3627.05 | −0.93 (z −16.7, 28/28) | 0.878 | 0.872 | +0.376 |
| 0.50 | 3626.56 | −1.42 (z −13.3, 28/28) | 0.854 | 0.839 | +0.498 |
| **0.75** | **3626.49** | −1.49 (z −9.6, 27/28) | 0.824 | 0.802 | +0.575 |
| 1.00 | 3626.85 | −1.13 (z −5.7, 26/28) | 0.791 | 0.766 | +0.623 |

**The optimum is interior at 0.75.** Section 13 recorded that no criterion this project had could
select the defensive shrinkage -- within-season stint MSE prefers the wrong end at 3.4 standard
errors, and the next-window on-court benchmark is flat then declines, so it runs to its grid
boundary. This one chooses. That is the methodological result of this section, independent of which
value it picks.

**And the consensus is monotone in the opposite direction.** It was read once, after `c_def` was
selected. Every step that improves held-out prediction costs rank agreement and adds archetype bias,
without exception, from 0.195 to 0.623 across the sweep.

### What that means, stated carefully

The two criteria are not in conflict; they measure different things, and the defensive box prior does
both at once. Rebounds and blocks are real events that correlate with a team's defensive outcome, so
crediting a big man with them predicts his lineups well. But the credit is partly his teammates' --
a defensive rebound is available because someone contested the shot -- so the lineup SUM is right
while the SPLIT is wrong. Predicting held-out points mostly rewards the sum. This is memory trap 2
("calibration on a lineup sum is blind to attribution") restated: leave-one-season-out is a large
improvement on the within-season version, since about half of a returning player's teammate-
possessions are with someone new, but the movers table shows only about half the gap is churn-
sensitive, so it remains substantially a lineup-sum test.

So the honest summary is that **`c_def` trades a forecasting product against a rating product**:

* If the deliverable is *predict what a lineup will do*, `c_def = 0.75` is right and measurably so.
* If the deliverable is *say who is good*, `c_def = 0` is right, and every anchor in
  `tests/test_vs_consensus.py` -- Robert Williams, the star guards, the bigness correlation -- is an
  attribution test that says so.

The project ships a player rating, so nothing is changed on this evidence. But it is now measured
rather than assumed, and the size of what is being given up is known: about 6% of the rating signal
in held-out prediction.
