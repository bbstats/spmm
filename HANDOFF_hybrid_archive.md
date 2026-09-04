# Handoff: build luck-adjusted possession results

Written after a session that diagnosed why the player ratings were tilted, fixed what could be
fixed, and established where the remaining headroom is, then updated after the session that shipped
the hybrid prior. `FINDINGS.md` sections 7-13 hold the detailed record;
`HANDOFF_diagnosis_archive.md` is the previous version of this file. This document is the short
version plus the plan for the next thing.

**State:** the hybrid prior is shipped (Part 1). The luck-adjusted target (Part 2) has not started.

---

## Part 1: what we established

### The validation target

`data/external/consensus.csv` — the project owner's own blend of the modern all-in-one metrics.
Columns `player_name, team, adj_offense, adj_defense, adj_overall`, points per 100 possessions, 582
players, one snapshot. Refresh with `python scripts/22_vs_consensus.py --refresh`.

**Use it as validation, never as something to fit to.** Choose every constant by an internal
criterion, then read the CSV once to score. Anything fit against it — even leave-one-team-out — is
an upper bound, not a candidate. We violated this once and had to withdraw a result.

It is also **not a ceiling**. The owner considers it beatable. Its own internal disagreement (the
`var` column) implies a reliability near 0.91, capping correlation with it around 0.956, but a
genuinely better metric could and should diverge from it in places. Rising CSV agreement is
evidence, not the goal.

Metric throughout: **Spearman rank correlation** between our rating (points per 100, the 2024-2026
window) and the matching CSV column, over the 475 players with 1000+ possessions who join by name.

### The ladder (`scripts/30_ladder.py`), and what shipped

| candidate | total | offense | defense |
|---|---|---|---|
| RAPM, no prior | 0.757 | 0.795 | 0.859 |
| box score alone, team-priced | 0.669 | 0.829 | 0.506 |
| box score alone, player-priced | 0.707 | 0.828 | 0.500 |
| prior-informed RAPM, team-priced prior | 0.786 | 0.868 | 0.785 |
| prior-informed RAPM, player-priced prior | 0.844 | 0.884 | 0.846 |
| + boosted correction — what shipped before | 0.787 | 0.852 | 0.772 |
| hybrid at ladder row 7's knobs | 0.853 | 0.884 | 0.859 |
| **hybrid at the shipped penalty — WHAT SHIPS NOW** | **0.896** | **0.879** | **0.888** |

**SHIPPED.** `config.yaml`'s `ratings_prior` block selects it, `windows.hybrid_beta` builds it, and
`scripts/08_ratings.py` passes it to the unchanged `player_ratings_table`. It is **one fit**, not
two: the box term is an offset `Xbox @ beta` with separate offensive and defensive columns, so
zeroing beta's defensive half IS "no defensive prior". +0.112 over what shipped before.

**Five of the six strict xfails flipped and are guards now** — defensive spread (2.13 → 1.23),
defensive agreement (0.755 → 0.888), archetype bias (+0.63 → +0.20), star guards, overall agreement.
The suite is 32 passed, 1 xfailed.

**The one that remains is an attribution defect, and it is the reason for the next piece of work.**
Pure on-court defensive RAPM still has Robert Williams 68th where the consensus has him 134th,
Jonathan Isaac 55th against 156th. Removing the box prior halved the tilt; no choice of prior
removes the rest, because the question is how one defensive possession is split among five players.

**A warning about the defensive shrinkage, before anyone re-tunes it.** The effective defensive
penalty is `lam * lam_ratio`. The ladder's internal criterion — the player's own next-window pure
on-court impact — is FLAT across the four weakest penalties and then declines, so it has no interior
optimum and cannot choose; ladder row 7's knobs are that criterion hitting its grid boundary. This
is the same failure as `scripts/26_which_loss.py`: the benchmark shares the estimate's blind spot,
so less shrinkage always looks fine. We kept `lam_plugin * lam_ratio_plugin` = 5271 because it is
the status quo, chosen before any of this. There is a measured but **unexploited +0.04** at an
effective penalty of 6000, and taking it requires an internal criterion that does not yet exist —
the consensus column has already been seen, so anything chosen now would be contaminated. See
`scripts/33_hybrid.py` and FINDINGS.md section 13.

### What is settled, with numbers

- **Joint RAPM and prior-informed RAPM are the same estimator.** The design uses window-level rates,
  so `X_box = Z R` and the joint model is `min ||y - Z(Rβ+u)||² + λ‖u‖²` over (β,u); the two-stage
  version is the same thing with β supplied. The only difference is how β is chosen — and the joint
  fit chooses it to minimise **stint** error.
- **Two different internal criteria are blind to the defensive shrinkage, in the same direction.**
  Held-out stint MSE (below) and the next-window on-court benchmark (FINDINGS.md section 13) both
  fail to see it, because both are built from the same margin data as the estimate. Assume any new
  internal criterion is blind until shown otherwise.
- **Stint MSE cannot select player-level parameters.** A stint row sees only the lineup *sum* of
  player effects, so reallocating credit between teammates barely moves it. Swept over the defensive
  prior weight, held-out stint MSE prefers the *wrong* end at 3.4 standard errors while moving only
  0.065% in absolute terms (`scripts/26_which_loss.py`). This is the root cause of everything that
  went wrong.
- **The joint β is right for the coefficient study and wrong as a ratings prior.** Same ranking as a
  player-level β (Spearman 0.95 offense, 0.87 defense) but the defensive amplitude is ~1.8x too
  wide. Conservation explains the aggregate amplitude (defensive rebounds survive into the lineup
  sum at 0.59, blocks 0.69, steals 0.90) but **not** the per-feature pattern — per-feature correction
  beats a flat scalar by only +0.003.
- **The joint fit is not the fixed point of a recursive prior-informed RAPM.** The first iteration
  moves β by 12.4%, and the recursion converges somewhere *worse* (total 0.786 → 0.765, defensive
  prior spread 1.88 → 2.10). It is a feedback loop: refitting β on `θ = Rβ + u` regresses the prior
  partly on itself, and because the ridge penalises `u` but not β, each turn launders shrunk
  residual into unshrunk prior. The joint fit's virtue is that it does **not** fully converge.
- **The box score has no defensive vocabulary.** Weighted R² of a player's 13 rates on his own
  on-court RAPM: **0.53 offense, 0.26 defense**. Pure RAPM beats our shipped defensive rating 0.877
  to 0.755.

### Measured to ~zero — do not spend time here again

| lever | gain |
|---|---|
| the boosted correction (`boost.py`) | 0 on total, −0.020 on offense |
| re-pricing the prior team → player, once amplitude is handled | ~0 |
| blend reweighting | ≤ +0.03, *and that is the fit-to-validation upper bound* |
| previous window's RAPM added to the current one | +0.008 |
| per-feature conservation correction vs a flat scalar | +0.003 |

### Fixed along the way

`08_ratings.py` was adding the boosted correction on top of a residual fit *without* it,
double-counting; it now passes it as `prior_offset` and the two ratings tables agree to 3e-13. (The
correction is out of the shipped rating entirely as of the hybrid — `ratings_prior.boost: false` —
but the offset machinery stays, because it is still the right way to use it if it is used.)
`chimeraboost` added to `pyproject.toml`. `site.py` theme substitution fixed (was emitting
`--text2:#0b0b0b2`). Dead code removed from `plots.py`. 27 tests pass; 6 strict xfails in
`tests/test_vs_consensus.py` encode the remaining defect and flip when it is fixed.

---

## Part 2: the next thing — luck-adjusted possession results

### Why this is the right next move, not a tangent

The causal chain behind every problem above was:

> the on-court signal is noisy → it gets shrunk hard → the box prior dominates by default →
> archetype bias

And the on-court signal is the **better** signal, decisively so on defense (0.877 against 0.506). So
the highest-value intervention is not a better prior or a better blend — both measured near zero —
it is **making the on-court signal less noisy**. Every unit of variance removed from `y` buys less
shrinkage, more weight on the thing that is already winning, and a smaller role for the thing
causing the bias.

It also attacks the identification problem from the only direction left. We cannot get more
possessions, but we can make each possession carry more signal.

### The method: expected points from four-factor RAPM

A possession's observed points contain shot-making and rebound variance the players on the floor did
not control. Replace observed points with **expected** points and the target keeps the skill, drops
the noise.

The possession does not end at the shot, though. A miss can be offensive-rebounded into another
attempt, and another. If you build expected points by summing the shots that *actually happened* you
have conditioned on the observed number of attempts, and whether the ball bounced back is exactly
the luck you were removing. So model the possession as a **geometric series** and never look at
whether the rebound occurred:

    V = x + m * r * V          ->      V = x / (1 - m * r)

      x  expected points from the attempt itself (expected FT% / 2P% / 3P%)
      m  probability the attempt misses
      r  OREB%, the probability the offense rebounds that miss

A worthwhile refinement is two stages, since putbacks convert well above average:

    V = x1 + m1 * r * V2,   V2 = x2 / (1 - m2 * r)

**Where `x`, `m` and `r` come from — this is the part to get right.** They are **not** league
averages conditioned on shot location. They are **lineup-specific expected rates from four-factor
RAPM**: run RAPM separately with eFG%, TOV%, OREB% and FT rate as the target, giving every player an
estimated effect on each factor on both sides of the ball. A possession's expected rate is then the
league baseline plus the ten on-court players' effects.

Location-only rates would give a five-guard lineup and a lineup with Steven Adams **the same
expected OREB%**. That is plainly wrong, and it is the same species of error as the box-score
defensive prior we spent this session removing: a rate that ignores who is actually on the floor,
applied as if it did not.

**It also resolves the skill-versus-luck question without hand-tuning.** An earlier draft of this
document proposed a manual asymmetry — shooter's own rate on offense, league rate on defense,
because defences control shot quality more than shot-making. Four-factor RAPM makes that
unnecessary: a lineup's expected eFG% already contains five offensive effects and five defensive
effects, and the fit decides how large each is. If defences really do have little eFG%-suppression
skill, the defensive eFG% effects come out small on their own. **Do not impose the asymmetry by
hand; let it be estimated.**

**The conservation subtlety, and why it does not bite here.** Four-factor RAPM has the same
attribution problem as everything else in this project: an offensive rebound goes to one of five
players, so splitting a lineup's OREB% among them is the badly-identified direction, and the
*individual* four-factor OREB ratings will be as shaky as our defensive prior was.

It does not matter for this use, and the reason is worth stating precisely: to build xPTS you need
the **lineup-level** expected rate, `Z @ theta_oreb`, not the individual split — and the lineup sum
is exactly the part RAPM estimates *well*. This is the mirror image of the session's main lesson.
**Use only the lineup sums; do not let the four-factor player ratings out as a product.**

**Do not turn this into a loop.** `scripts/32_recursion.py` showed that recursion in this codebase
converges somewhere worse, because refitting a prior on a quantity containing that prior launders
shrunk residual into unshrunk prior. What is proposed here is not that — the four-factor RAPMs are
fit on *different targets* than the points RAPM, so there is no direct self-reference. Keep it that
way: **fit four-factor once, freeze it, build xPTS, then fit the points RAPM. One pass.** Before
ever closing that loop, re-read script 32 and check the convergence behaviour explicitly.

**What we cannot replicate.** The owner's system has inputs ours does not — no defender distance or
openness, so our shot-quality term is coarser. A reason to validate carefully, not to skip it.

### The data, and where to intervene

`data/raw/pbp/{season}/` has, for all 30 seasons, one row per event with:

    isFieldGoal, shotResult (Made/Missed), shotValue (2 or 3), shotDistance,
    xLegacy, yLegacy (shot coordinates), actionType, subType, personId, teamId, period, clock

Verified populated for 1997, 2000, 2005, 2010, 2014, 2016, 2020 and 2026. Rebounds are
`actionType == "Rebound"`, offensive when the rebounder's `teamId` matches the shooter's.

`possessions()` in `src/eracoef/stints.py` builds its record dict around line 290 carrying only
`points`. It needs per-possession counters — FGA, FGM, FG3M, OREB, DREB, TOV, FTA, FTM — added
there, with `stints()` aggregating them alongside `pts_h`/`pts_a` (around line 465). The four RAPM
fits then reuse `MixedModelRAPM` unchanged: design matrix, cross-fitting and lambda path all carry
over, only the target column changes.

**Carry expected points alongside actual, never instead of.** Add `xpts_h` / `xpts_a` and leave
`pts_h` / `pts_a` untouched. The ratings move to xPTS; actual points stay because (1) they are the
ground truth everything is validated against, and (2) the coefficient drift study should stay on
points that were really scored.

### Build order

1. **Per-possession counters** into `possessions()` and `stints()`. Nothing clever, but everything
   below depends on it.
2. **The four-factor RAPMs.** Four fits, existing machinery, target column swapped.
3. **Free throws.** `FTA x expected FT%` — zero ambiguity, roughly a fifth of all scoring, and the
   simplest possible end-to-end check that the pipeline works.
4. **The shot term and the geometric continuation.** Expected conversion and OREB% from the
   four-factor lineup sums, closed as `V = x / (1 - m*r)`.
5. **Later, if 1-4 pay off.** And-one luck, opponent free-throw luck. Diminishing returns.

**A note on leakage.** Wherever a player's own rate feeds a term later used to score him, it must be
cross-fitted or leave-one-game-out. **Reuse the pattern in `exposure.py`** rather than writing a new
one; that machinery exists because full-season rates inflated the three-point coefficient by more
than seven standard errors.

### Reference numbers: league OREB% by location

**Not the method** — the four-factor lineup sums are. Keep these as a sanity check on the fitted
rates, and as a fallback if the four-factor fits misbehave. Measured on 60 games of 2026:

| shot distance | 0-3 | 4-8 | 9-16 | 17-22 | 23-25 | 26-30 | 31+ |
|---|---|---|---|---|---|---|---|
| OREB% | 30.4 | 31.0 | 22.1 | 20.0 | 22.0 | 20.7 | 15.2 |

2PT 28.1%, 3PT 21.4%, overall 24.4%. At league averages the continuation multiplier `1/(1 - m*r)` is
about **1.15** — a possession is worth roughly 15% more than its first attempt's expected value, and
the multiplier is fairly stable across shot types because `m` and `r` move opposite each other. If a
fitted version departs far from 1.15 on average, something is wrong.

**Linking a miss to its rebound.** Only 89.4% of missed field goals are immediately followed by a
`Rebound` event; the rest are blocks (the block event intervenes), period ends and tip sequences. So
scan forward a few events rather than taking `shift(-1)`, and treat end-of-period misses as terminal
rather than silently counting them as defensive rebounds — that would bias OREB% down.

### How to validate — this is the good part

**The target is xPTS. The validation is that xPTS predicts actual PTS out of sample.** That is the
whole test, and it needs no external data and no model of anyone else's.

Use **interleaved** splits, not chronological ones: every other possession, or every other game as a
workable proxy. Interleaving holds lineups, roster, age and context fixed across the two halves, so
the only thing that differs is which way the bounces went. A chronological split confounds the test
with aging and trades.

    fit on half A with xPTS as the target  ->  predict half B's ACTUAL points
    fit on half A with PTS  as the target  ->  predict half B's ACTUAL points

If the xPTS-trained model predicts real, observed points better, xPTS is the better target. Actual
points in held-out interleaved games are **ground truth** — not another model's output, not our own
next-window estimate. It is the cleanest criterion available anywhere in this project, and it is the
answer to the "what is the right loss" question that earlier sections could only hedge on.

**But be precise about what it validates.** Interleaved halves validate the **target** — is xPTS a
less noisy measurement of the same underlying quantity. Because the same lineups appear in both
halves, it does **not** validate **attribution** — whether credit is split correctly among the five
players on the floor. Two different axes:

| question | criterion |
|---|---|
| is xPTS a better measurement than PTS? | interleaved halves predicting actual points |
| is credit split correctly among teammates? | the consensus CSV, the named anchors, next-window RAPM |

Run both. Do not let a win on the first be reported as a win on the second — that conflation is the
mistake that produced the original defect.

**Do not compare held-out MSE across different targets.** xPTS has lower variance than PTS by
construction, so a model fit on it will show lower error against *itself* for reasons unrelated to
quality. Always score against actual points.

**Secondary checks, in order.** Does rating stability between adjacent windows rise? Does the fitted
`lam_plugin` fall and the on-court residual's spread rise relative to the prior's? Both are the
mechanism working as intended and both are directly observable. Then the CSV, read once. Then the
anchors in `tests/test_vs_consensus.py`.

### Scripts you will want

    scripts/22_vs_consensus.py    fetch/refresh the CSV, join, deltas, diagnosis
    scripts/26_which_loss.py      the stint-MSE-cannot-see-this demonstration
    scripts/29_scorecard.py       every candidate, both metrics, paired bootstrap
    scripts/30_ladder.py          the ladder above, internal selection + external validation
    scripts/32_recursion.py       the recursion experiment
    scripts/33_hybrid.py          the defensive penalty path under both criteria; the shipped board
    scripts/34_hybrid_verify.py   single-fit vs spliced hybrid, and the scorecard against 3 baselines
    tests/test_vs_consensus.py    10 guards + 1 strict xfail

### One thing still not done

**Measure the estimand mismatch.** We emit a 3-season average; the CSV is a snapshot, and our own
RAPM correlates only 0.544 between adjacent windows. Fit a single-season or recency-weighted variant
and re-score. This says whether the remaining headroom is really ~0.10 or nearer 0.04, and therefore
how much the luck work has to beat. Not blocking, but it sizes the prize.

(The other item here — ship the hybrid — is done. See above.)

### One standing warning

Across this session I asserted a mechanism confidently and was wrong three times — the archetype
tilt was real when I said it was not, the booster was worthless when I said it was valuable, and
player history was negligible when I expected it to be the big win. Every one of those was caught by
measuring rather than reasoning. Measure first here too.
