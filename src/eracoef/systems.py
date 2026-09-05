"""The named systems the holdout CLI (scripts/45_holdout.py) and the parallel runner can build.

A registry function rather than a module-level dict so that a spawned worker process can rebuild
systems by name (closures such as the role-prior offset do not pickle), and so that the same names
mean the same thing in every script.

    rapm            no box prior
    pi              the team-priced prior (prior-informed RAPM)
    hybrid          player-priced offense, no defensive prior
    hybrid_xft      hybrid on the free-throw-adjusted target
    hybrid_c<c>     hybrid offense with the team-priced defensive prior scaled by c
    hybrid_<t>      hybrid on a shooter-level target t (xshoot.TARGET_REGISTRY); split_<t> = offense only
    def3_p<n>       offense from hybrid_xft, defense from the opponent-3PM-replaced target (the board)
    spm             the role prior alone on both sides (APM -> Simple SPM -> ridge), no box prior
    mspi_resid        role prior + the boosted box prior on both sides
    mspi_resid_o      role prior on both sides + the boosted box prior on offense only
    rankmap_<s>     any of the above with the leave-one-season-out rank map (needs a rank table)
"""
from __future__ import annotations

import pandas as pd

from .holdout import (PluginSystem, RankMappedSystem, ReplacementSystem, SplitSystem, beta_hybrid, beta_mixed, beta_none,
                      beta_team)


def registry(cfg, rankmap=None) -> dict:
    S = {
        "rapm": PluginSystem("rapm", beta=beta_none),
        "pi": PluginSystem("pi", beta=beta_team),
        "hybrid": PluginSystem("hybrid", beta=beta_hybrid),
        "hybrid_xft": PluginSystem("hybrid_xft", target="xpts_ft", beta=beta_hybrid),
        "pi_xft": PluginSystem("pi_xft", target="xpts_ft", beta=beta_team),
    }
    for c in (0.25, 0.5, 0.75, 1.0):
        S[f"hybrid_c{c:g}"] = PluginSystem(f"hybrid_c{c:g}", beta=beta_mixed(c))
        S[f"hybrid_xft_c{c:g}"] = PluginSystem(f"hybrid_xft_c{c:g}", target="xpts_ft", beta=beta_mixed(c))
    try:                                                   # the shooter-level targets, once built
        from . import xshoot
        for name, target in xshoot.TARGET_REGISTRY.items():
            S[f"hybrid_{name}"] = PluginSystem(f"hybrid_{name}", target=target, beta=beta_hybrid)
            S[f"split_{name}"] = SplitSystem(f"split_{name}", offense=S[f"hybrid_{name}"], defense=S["hybrid"])
        # defensive targets: offense from the free-throw system, defense from a fit on the target
        for name, target in xshoot.DEFENSE_TARGETS.items():
            S[f"def3_{name[6:] or 'p0'}"] = SplitSystem(f"def3_{name[6:] or 'p0'}", offense=S["hybrid_xft"],
                                                        defense=PluginSystem(f"hybrid_{name}", target=target, beta=beta_hybrid))
        # the role-prior chain (spm.py, gbdt_prior.py): no linear box prior, the offset carries everything;
        # offense on the free-throw target, defense on the opponent-3PM-replaced one, like the board
        from .spm import chain_offset

        def chain(name, sides, mode="residual", scale=1.0, target="rapm1"):
            o = PluginSystem(f"{name}_o", target="xpts_ft", beta=beta_none,
                             offset=chain_offset(sides, mode, scale=scale, target=target))
            d = PluginSystem(f"{name}_d", target=xshoot.DEFENSE_TARGETS["x3def"], beta=beta_none,
                             offset=chain_offset(sides, mode, scale=scale, target=target))
            return SplitSystem(name, offense=o, defense=d)

        S["spm"] = chain("spm", ())
        S["mspi_resid"] = chain("mspi_resid", ("O", "D"))                    # SPM + GBDT on the residual beyond it
        S["mspi_resid_o"] = chain("mspi_resid_o", ("O",))
        S["mspi"] = chain("mspi", ("O", "D"), mode="full")     # the GBDT (rates + season + role) alone
        S["mspi_o"] = chain("mspi_o", ("O",), mode="full")     # ... on offense; SPM on defense
        for sc in (1.25, 1.5, 2.0):                            # the GBDT prior scaled before the ridge
            S[f"mspi_s{sc:g}".replace(".", "")] = chain(f"mspi_s{sc:g}".replace(".", ""), ("O", "D"), mode="full", scale=sc)
        S["mspi_apm"] = chain("mspi_apm", ("O", "D"), mode="full", target="apm")   # trained on unshrunk APM
        for sc in (1.25, 1.5):
            S[f"mspi_apm_s{sc:g}".replace(".", "")] = chain(f"mspi_apm_s{sc:g}".replace(".", ""), ("O", "D"), mode="full",
                                                          scale=sc, target="apm")

        # the APM-trained offense with the RAPM_1-trained defense (each side's calibrated version)
        S["mspi_mix"] = SplitSystem("mspi_mix", offense=S["mspi_apm"], defense=S["mspi"])
        # a replacement level for players the block never saw, on the board and on the chains
        for n in ("def3_p0", "mspi", "mspi_apm", "mspi_mix"):
            S[f"{n}_rep"] = ReplacementSystem(f"{n}_rep", S[n])

        # garbage time down-weighted in the FIT only (the held-out scoring never changes): the board and
        # the multi-stage chain at gt_weight 0.5 and 0, so the comparison holds the weighting fixed
        from .windows import build_window

        def gt_target(base: str, w: float):
            def target(seasons, cfg, wd_pts):
                if base == "x3def":
                    wd05 = build_window(seasons, cfg, gt_weight=w)
                    return xshoot.DEFENSE_TARGETS["x3def"](seasons, cfg, wd05)
                return build_window(seasons, cfg, gt_weight=w, target=base)
            target.__name__ = f"{base}_gt{w:g}"
            return target

        for w in (0.5, 0.0):
            tag = f"gt{w:g}".replace(".", "")
            o_t, d_t = gt_target("xpts_ft", w), gt_target("x3def", w)
            S[f"def3_{tag}"] = SplitSystem(f"def3_{tag}", offense=PluginSystem(f"hybrid_xft_{tag}", target=o_t, beta=beta_hybrid),
                                           defense=PluginSystem(f"hybrid_x3def_{tag}", target=d_t, beta=beta_hybrid))
            S[f"mspi_{tag}"] = SplitSystem(f"mspi_{tag}",
                                           offense=PluginSystem(f"mspi_{tag}_o", target=o_t, beta=beta_none,
                                                                offset=chain_offset(("O", "D"), "full")),
                                           defense=PluginSystem(f"mspi_{tag}_d", target=d_t, beta=beta_none,
                                                                offset=chain_offset(("O", "D"), "full")))
    except ImportError:
        pass
    if rankmap:
        rank_table = pd.read_parquet(rankmap)
        have = set(rank_table.system)
        for n in list(S):
            if n in have:
                S[f"rankmap_{n}"] = RankMappedSystem(f"rankmap_{n}", S[n], rank_table)
        if "rankmap_def3_p0" in S:                     # the board as it ships, plus the replacement level
            S["rankmap_def3_p0_rep"] = ReplacementSystem("rankmap_def3_p0_rep", S["rankmap_def3_p0"])
    return S
