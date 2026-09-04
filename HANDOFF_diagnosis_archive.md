# Handoff: the ratings diagnosis, corrected twice

**Read this first.** An earlier version of this document concluded that the board's tilt toward big
men "is not a bias" and that the ratings were sound. **That was wrong**, and the reason it was wrong
matters more than the conclusion: it was validated against our OWN next-window on-court RAPM, a
benchmark built from the same margin data with the same blind spots. It could not catch a shared
error and it did not.

Scored against an external blend of the modern all-in-one metrics
(`data/external/consensus.csv`, `scripts/22_vs_consensus.py`), the 2024-26 board is badly tilted:

| | our rank | consensus rank |
|---|---|---|
| Robert Williams III | 25 | 134 |
| Jusuf Nurkic | 21 | 166 |
| Jonathan Isaac | 18 | 156 |
| Luke Kornet | 12 | 75 |
| Day'Ron Sharpe | 11 | 82 |
| Trae Young | 313 | 79 |
| LaMelo Ball | 95 | 10 |
| Devin Booker | 79 | 10 |

The rule that generated the old, wrong conclusion is now in the test suite as
`tests/test_vs_consensus.py`, with the defect encoded as six strict xfail tests that flip to
passing when it is fixed.

## The defect in one table

| | rank agreement | our spread | consensus spread | ratio |
|---|---|---|---|---|
| offense | 0.851 | 2.05 | 1.96 | **1.05** |
| defense | 0.755 | 2.33 | 1.09 | **2.13** |
| total | 0.784 | 3.42 | 2.44 | 1.40 |

**Offense is fine. Defense is twice as wide as it should be.** The gap correlates +0.59 with a
bigness index on defense and only +0.17 on offense, and the two box-score columns that carry it are
offensive rebounds (+0.55) and blocks (+0.51) — exactly the stats the box score can see and the ones
big men accumulate.

Note what the consensus says about the shape of the game: individual **defensive** impact varies
about **half** as much as offensive impact (spread 1.09 against 1.96). We produce defensive spread
slightly *larger* than offensive (2.33 against 2.05). The ratio is inverted.

## Yes, we penalise defense differently — and that is backwards

`config.yaml` ships `lam_ratio_plugin: 0.2872`, so the defensive penalty is 5,271 against 18,352 on
offense: **defense is shrunk 3.5x less**. The consequence is visible in the pieces:

| piece | offense spread | defense spread | D over O |
|---|---|---|---|
| box prior | 1.92 | 2.01 | 1.04 |
| correction | 0.57 | 0.32 | 0.56 |
| on-court residual | 0.30 | 0.99 | **3.31** |
| the rating | 2.05 | 2.33 | 1.13 |

That choice was made by REML, which put defensive player variance about 3.5x offensive. REML is
fitting *stint margin*, and it cannot separate "this lineup defended well" from "this player
defends well" — defense is far more scheme- and unit-driven, so the ridge happily hands unit-level
variance to individuals. `FINDINGS.md` section 1 called this "what the data says, not a tuning
artifact". Against an external benchmark it is a tuning artifact.

**But the penalty ratio is the second problem, not the first.** Sweeping it all the way to 1.0
moves the total rank agreement only from 0.784 to 0.796. Here is why:

| | prior correlates with consensus | residual correlates | prior spread | residual spread |
|---|---|---|---|---|
| offense | **+0.87** | +0.34 | 1.92 | 0.30 |
| defense | +0.53 | **+0.66** | 2.01 | 0.99 |

On defense the on-court residual is the **better** signal and the box prior the **worse** one, yet
the prior carries twice the spread. The penalty ratio only scales the residual, so turning it up
throws away the better signal. What needs to shrink is the defensive **box prior** — and no lambda
in this model can do that, because the prior enters the rating at weight exactly 1.0 by
construction. That structural fact was identified in the previous round; what was wrong then was the
conclusion that it did not matter.

## What fixes it

Two principles guided this: **RAPM on possessions is the basis, informed by a prior**, and **adjust
for sample size and luck as smartly as you can**. The first is already satisfied — the plug-in fit
carries the box prior as the offset `Z @ prior`, so a player is shrunk toward his box expectation
rather than toward zero. The second was being violated: the prior enters at weight exactly 1.0 on
both sides, asserting that the box score measures defense as reliably as offense.

**The obvious internal fix fails, and the failure is the most important result here.** Let the stint
regression estimate the trust itself, as two free unpenalized columns (`free_prior_scale`, now in
`estimator.py`; `scripts/24_free_prior_scale.py`). It returns **1.12 on offense and 1.06 on
defense** — trust the box score *more* — and moves Robert Williams from 25th to 12th.

The reason is an identification limit, not a bug. **The stint likelihood depends on players only
through lineup sums, and the defensive prior's error is a reallocation of credit between teammates,
which leaves the lineup sum almost unchanged.** The conservation numbers say it exactly — the
fraction of a player's own rate difference that survives into the lineup sum:

| defensive rebounds | offensive rebounds | blocks | steals | missed twos |
|---|---|---|---|---|
| 0.59 | 0.60 | 0.69 | 0.90 | 0.88 |

The three stats the defensive prior is built from are the three most conserved. A defensive rebound
goes to exactly one of five players, so the team total is nearly fixed and the individual rate
mostly records *who collected it*. Assembled, the offensive prior survives into lineup variation at
1.006 and the defensive prior at 0.787.

**So the defensive coefficients are right at the level they were estimated (lineups) and wrong at
the level they are applied (players).** A team that blocks more does allow fewer points; that does
not mean the man who blocked it is the man responsible. No internal check can price this, because
the data is nearly invariant to it. Something external has to set the weight — which is why the
consensus benchmark is not optional.

### The structural fix

Scale the defensive half of the plug-in offset by `c` so the residual is refit knowing the prior is
trusted less, and sweep the penalty ratio with it (`scripts/25_defensive_prior.py`). Beta is
untouched, so the coefficient study does not move. Scored against the consensus, 2024-26:

| `c_def` | ratio | total | defense | defensive spread | bigness bias |
|---|---|---|---|---|---|
| 1.00 (ships) | 0.287 | 0.779 | 0.750 | 2.12x | +0.635 |
| 0.30 | 0.287 | 0.868 | 0.861 | 1.43x | +0.401 |
| 0.10 | 0.287 | 0.887 | 0.885 | 1.29x | +0.254 |
| **0.00** | **0.50** | **0.890** | **0.886** | **0.93x** | **+0.063** |
| 0.00 | 0.287 | 0.892 | 0.887 | 1.23x | +0.162 |

**The optimum is zero.** The best defensive rating uses no box score at all — pure on-court RAPM.
It beats the post-hoc three-piece reweight (0.864) while being one structural parameter instead of
six fitted ones, and it drives the archetype bias essentially to nothing (+0.063 at ratio 0.5).

This is not circular. If the consensus were simply RAPM in disguise, the on-court residual would
beat the box prior on offense too. It does not — predicting the consensus from each piece alone:

| | box prior alone | on-court residual alone |
|---|---|---|
| offense | **0.829** | 0.365 |
| defense | 0.506 | **0.672** |

The box score carries real offensive information the margin lacks, and on defense the relationship
inverts. That asymmetry is the finding, and it is exactly what the conservation table predicts.

At `c_def = 0.0, ratio = 0.5`: Robert Williams 13 -> 76 (consensus 134), Nurkic 27 -> 75 (166),
Jonathan Isaac 18 -> 59 (156), Trae Young 299 -> 142 (79), Curry 72 -> 13 (10.5), Booker 89 -> 28
(10.5). The top 20 becomes Shai, Jokic, Luka, Giannis, Kawhi, Butler, Embiid, Wembanyama, Mitchell,
Tatum.

### Caveats before shipping this

- `c_def` was chosen on one window against one external benchmark. It is a single parameter with a
  monotone gradient to the boundary and an independently established mechanism, so it is not a
  fitting artifact — but it has not been validated on another window, and the consensus is a
  current-season blend while our window pools three seasons.
- Dropping the defensive prior will make **held-out stint MSE slightly worse**, because the prior
  does explain lineup-level variance. That is the point: stint MSE is the wrong loss for ranking
  players, and optimising it is what produced this defect.
- Offense should keep its prior at 1.0. Nothing here argues for touching it.

## What is still true from the previous round

- The coefficient drift study is untouched by all of this and remains the deliverable.
- Three-season windows beat five.
- Squared lineup-sum terms fail out of sample.
- The double-count fixed in `08_ratings.py` (`FINDINGS.md` section 8) was real and is fixed.
- The four correctness cleanups landed.

## What is now retracted

- "The big-tilt is not a bias." It is.
- "The total board is sound." It is not; rank agreement with the consensus is 0.784 and should be
  about 0.86.
- "No constant should change." The O/D penalty ratio should move toward 1.0, and more importantly
  the defensive prior needs a weight below 1.0, which the model currently cannot express.
- The next-window criterion should not be used as a primary benchmark again. It is not neutral.

---

## The project in one paragraph

Thirty NBA seasons of play-by-play turned into stints, then one ridge regression per three-season
window: box-score exposures as unpenalized fixed effects beside ridge-penalized player offense and
defense effects, solved by Henderson's mixed model equations. **The deliverable is the coefficient
drift study** — how the value of a 3PM, an ORB, a block changed from 1997 to 2026. Player ratings
are a by-product that fall out of the same fit. The coefficient study is finished and validated and
nothing below touches it.

## What the previous handoff claimed, and what the evidence says

It claimed the rating over-weights the box prior by about 1.7x, that `lam_plugin` was the one knob
that fixes it, that this is why bigs sweep the 2024-26 board, and that the booster should probably
then be deleted. Four claims. All four fail.

**1. `lam_plugin` cannot re-blend the rating.** The rating is `prior + u`. The plug-in fit holds
beta fixed and prices the assembled prior at exactly 1.0; `lam_plugin` only controls how much `u`
is added on top. Lowering it does not take weight off the prior, it adds weight to the residual.
At the criterion's own optimum the prior still carries 73% of the rating's variance, not the 40%
the previous handoff expected. The "alpha" it reported was a blend of prior with a *zero-prior
RAPM*, which is a different object from `u`, and the pipeline has no knob that traverses it.

**2. Re-blending buys almost nothing, by any route.** Scoring against the previous handoff's own
criterion — window w's rating against the same player's window w+1 pure on-court RAPM, 2133 player
pairs with 3000+ possessions in both:

| | corr with next window |
|---|---|
| shipped (`lam_plugin` 18352, ratio 0.2872) | 0.5739 |
| best in the shipped `prior + u` family (lam 10899, ratio 0.18) | 0.5807 |
| best with the prior weight `c` *freed as well* (`c·prior + u`) | 0.5809 |
| the ceiling: regress the target on (prior, u) with free coefficients | 0.5811 |

The whole re-blending question is worth **+0.007**, and freeing the prior weight adds 0.0002 on top
of just moving lambda. The gain is distinguishable from zero (bootstrap over window pairs, 95% CI
[+0.004, +0.010]) and practically inert: Spearman 0.985 between the two boards, 17/20 top-20
overlap, and the big share of the 2024-26 top 20 goes **up**, 0.70 to 0.75.

**3. The board's tilt toward bigs is not a bias.** This is the decisive test, and the previous
handoff coded it in `scripts/15_investigate.py` but never reported its result. Put the shipped
rating in a weighted regression predicting the next window's pure on-court RAPM — a target with no
box score anywhere in it — and add a bigness term. If the rating over-credits bigs the term is
negative. It is **+0.045, z = +4.7**. Positive.

Standardising each variable by its own spread, so the tilts are comparable:

| tilt toward bigs, per standard deviation | total | offense | defense |
|---|---|---|---|
| next window's pure on-court RAPM (the evidence) | +0.042 | -0.057 | +0.091 |
| the shipped rating | +0.025 | **-0.145** | **+0.188** |
| the box prior alone | +0.021 | -0.148 | +0.213 |

**The total board is tilted toward bigs less than the on-court margin independently says it should
be.** Thirty seasons of on-court evidence, with no box score in it, likes bigs. So does the board.
That complaint is answered: there is nothing to fix there.

**4. The booster is the most valuable piece, not the one to delete.** On the same criterion it is
worth **+0.0129**, which is 1.9x the entire re-blending question, and once it is in, re-tuning
`lam_plugin` is worth +0.003 and the optimum moves *up* from 18573 rather than down.

*(Corrected 2026-09-04. The first measurement of this said +0.0174 and 2.6x. It added the
correction on top of a residual fit without it, which is the bug described under "the two ratings
tables" below; +0.0129 is the value on the convention that actually ships. I expected the fix to
raise this number and it lowered it.)*

## What is actually wrong: the per-side split, and it cancels in the total

Read the table in claim 3 by column. The rating reproduces the sign of the on-court evidence on
both sides but roughly **doubles the gradient**: -0.145 against -0.057 on offense, +0.188 against
+0.091 on defense. As archetype dummies with guards as the reference, the rating already in:

| | offense | defense | total |
|---|---|---|---|
| wing | +0.165 (z 4.3) | -0.033 (z -0.6) | +0.299 (z 4.3) |
| big | +0.226 (z 5.5) | -0.280 (z -4.2) | +0.289 (z 4.0) |

The offensive rating under-credits bigs, the defensive rating over-credits them by almost exactly
as much, and the two cancel. The mechanism is the one the previous handoff correctly identified:
the box score cannot see perimeter defense, so the defensive prior is built from blocks, defensive
rebounds and steals, which bigs accumulate. **The site shows offense and defense as separate views,
so this is a real defect even though the headline board is sound.**

Note the total column: wings and bigs are *equally* under-credited relative to guards, +0.299 and
+0.289. The one-dimensional bigness story was never right — the mispricing is U-shaped, and a
linear bigness term cannot see it. The group the total board over-credits is guards.

The booster already moves this in the right direction and is the natural tool for it, since a
nonlinear function of the box rates is exactly what expresses "this rate profile has a defensive
prior that is too high". Its mean correction by tercile is +0.114 offense for wings against -0.089
and -0.097 for guards and bigs: the U shape, learned rather than imposed.

| dummy, guards the reference | without booster | with booster |
|---|---|---|
| offense, wing | +0.165 | **+0.095** |
| offense, big | +0.226 | **+0.186** |
| defense, big | -0.280 | **-0.227** |

It shrinks all three and eliminates none.

## Recommended plan

1. **Change no constants.** `lam_plugin` 18351.8 and ratio 0.2872 sit on the plateau of the
   player-level criterion once the booster is in (0.5912 against a 0.5941 argmax that is flat from
   18573 to 53940). Re-tuning them was the previous plan's headline and it is not worth doing.
2. **Keep the booster.** Delete `boost.py` and it costs 0.017 on the player-level criterion and
   gives back the only correction that touches the real bias.
3. **If you want to fix the per-side split**, that is new work and the target is explicit: drive
   the offense and defense archetype dummies in the table above to zero without moving the total.
   Worth deciding whether it is in scope — the coefficient study does not need it, and the total
   board does not need it either.
4. **Leave the coefficient study alone.** It is the actual deliverable, it is validated, and none
   of this touches it. Beta was held fixed at the same cross-fitted `lam_beta` throughout.
5. Still open and untouched: the empirical-Bayes padding constant `k` is estimated within a season
   and applied to a three-year unit, so it is probably too small (`FINDINGS.md` section 6).

## What is settled — do not re-litigate without new evidence

- **Player-window (3-year) units**, constants `lam_beta` 10149.5 / ratio 0.5, `lam_plugin` 18351.8 /
  ratio 0.2872, `pad_target` poss_conditional. Coefficients barely moved under the change.
- **Three-season windows beat five.** Held-out MSE 3624.5 vs 3624.9, rating stability 0.714 vs 0.676.
- **Squared lineup-sum terms (diminishing returns) are rejected** — unanimous in sample, 0.3 bp
  worse out of sample, rank correlation 0.998 with the board.
- **The offense/defense split is identified** (corr of a player's own O and D residual is +0.09).
- **The blend is not the problem** (this document, claims 1 and 2).
- **The big-tilt of the total board is not a bias** (claim 3).

## Caveats on the tests above

- The criterion's target is a single-window pure on-court RAPM, which is noisy; that attenuates
  every correlation but does not favour one candidate over another. Conclusions are about
  differences, not levels.
- A player's `u` in window w and his target in w+1 share teammate contamination if he stays put.
  Split on whether the dominant team changed, the optimum barely moves (best lam 8349 for movers
  against 10899 for stayers), so this is not driving the result.
- The booster is cross-fitted on player id, so a player never scores himself in any window. His
  former teammates can still be in its training set; that channel is second-order and untested.

## Read this first: outputs/diagnostic.html

`python scripts/21_diagnostic.py` builds it in seconds from cached tables. Six figures, hover for
player names, no Greek letters anywhere: what a rating is made of, the flat shrinkage sweep, the
tilt-toward-bigs comparison, who gets mispriced by name, the 2024-26 offense/defense trade, and
what the correction does. It is the argument of this document in a form you can check by eye.
It is a working artifact and is written to `outputs/`, not published to `docs/`.

## State of the tree

Nothing is committed. `config.yaml` is untouched and no modelling constant changed. Four
correctness fixes landed: `chimeraboost` added to `pyproject.toml`, the theme-token substitution in
`site.py` fixed (it was emitting an invalid colour), `plots._page` and `plots.index_page` removed as
dead, and the ratings double-count in `08_ratings.py` fixed (see `FINDINGS.md` section 8). 22 tests
pass and `docs/index.html` is regenerated.

    scripts/04_fit_all.py       coefs.parquet, base plus six robustness runs
    scripts/05_playoffs.py      the playoff block
    scripts/08_ratings.py       player_ratings.parquet (one row per player-window)
    scripts/07_plots.py         docs/index.html
    scripts/10_boost.py         the pooled LRBoost; caches its panel to data/cache (gitignored)
    scripts/11_gate.py          calibration gate, boosted vs linear prior
    scripts/12_window_length.py three-season against five-season
    scripts/13_boost_oos.py     held-out MSE, boosted vs linear
    scripts/14_boost_null.py    permutation null for the booster shrinkage
    scripts/15_investigate.py   the diagnostics behind the previous version of this document
    scripts/16_tune_plugin.py   the (lam, ratio) grid against the next-window player-level criterion
    scripts/17_prior_weight.py  frees the prior weight too, and finds the ceiling
    scripts/18_board_effect.py  is the gain real, and does the board move (bootstrap + top 20)
    scripts/19_archetype_bias.py  the decisive test: is the big-tilt a bias or is it evidence
    scripts/20_boost_archetype.py does the booster fix the per-side bias, and is it worth keeping
    scripts/21_diagnostic.py    outputs/diagnostic.html, the six figures above
    scripts/diagnostic_page.py  the page template 21 renders through

Scripts 11, 13 and 14 need `data/cache/boost_panel.parquet`, which `10_boost.py` writes. Scripts
17-21 read `outputs/tune_players.parquet` and `outputs/tune_u.parquet`, which `16_tune_plugin.py`
writes (about 3 minutes); `16` takes `--rescore` to skip the rebuild.

Note that scripts 16-20 were written before the double-count in section 8 was found, so the
"boosted" numbers in `20_boost_archetype.py` use the old convention. `21_diagnostic.py` re-measures
them correctly and is the one to trust; the corrected figures are in this document and in
`FINDINGS.md` sections 7 and 8.
