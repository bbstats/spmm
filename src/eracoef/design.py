"""Window -> sparse design matrix.

X = [Z_O | Z_D | F | game_idx] as CSR, plus y, sample_weight, groups and a DesignSpec.
After LOOExposure, X gains 2 * n_feat dense columns [Xbox_O | Xbox_D] at the end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURES = ["fg3m", "fg3_miss", "fg2m", "fg2_miss", "ftm", "ft_miss",
            "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
HOME_SLOTS = [f"h{i}" for i in range(1, 6)]
AWAY_SLOTS = [f"a{i}" for i in range(1, 6)]
STINT_COLUMNS = ["game_id", "season", "phase", "series_id", "period", *HOME_SLOTS, *AWAY_SLOTS,
                 "poss_h", "poss_a", "pts_h", "pts_a", "margin_h", "frac_rem", "is_gt", "neutral"]


def ps_key(player_id, season) -> np.ndarray:
    """Integer key for (player_id, season)."""
    return np.asarray(season, dtype=np.int64) * 100_000_000 + np.asarray(player_id, dtype=np.int64)


@dataclass
class DesignSpec:
    """Column layout of the design matrix and the player-season id behind each Z column."""
    n_ps: int
    seasons: list
    season_of_ps: np.ndarray          # (n_ps,) index into `seasons`
    ps_table: pd.DataFrame            # ps_idx, player_id, season
    f_names: list
    features: list
    col_groups: dict = field(default_factory=dict)   # name -> Z column indices (0..2*n_ps)

    @property
    def zo(self) -> slice:
        return slice(0, self.n_ps)

    @property
    def zd(self) -> slice:
        return slice(self.n_ps, 2 * self.n_ps)

    @property
    def z(self) -> slice:
        return slice(0, 2 * self.n_ps)

    @property
    def f(self) -> slice:
        return slice(2 * self.n_ps, 2 * self.n_ps + len(self.f_names))

    @property
    def game_idx_col(self) -> int:
        return 2 * self.n_ps + len(self.f_names)

    @property
    def n_base(self) -> int:
        return self.game_idx_col + 1

    def box_o(self, n_feat: int) -> slice:
        return slice(self.n_base, self.n_base + n_feat)

    def box_d(self, n_feat: int) -> slice:
        return slice(self.n_base + n_feat, self.n_base + 2 * n_feat)

    @property
    def is_po_col(self) -> int:
        return 2 * self.n_ps + self.f_names.index("is_po")

    def f_col(self, name: str) -> int:
        return 2 * self.n_ps + self.f_names.index(name)

    @property
    def n_seasons(self) -> int:
        return len(self.seasons)

    def z_season(self) -> np.ndarray:
        """Season index of every Z column (O block then D block)."""
        return np.concatenate([self.season_of_ps, self.season_of_ps])

    def season_cols(self) -> list:
        """Per season, the Z columns (O and D) of that season's player-seasons."""
        out = []
        for s in range(self.n_seasons):
            idx = np.flatnonzero(self.season_of_ps == s)
            out.append(np.concatenate([idx, idx + self.n_ps]))
        return out


@dataclass
class WindowData:
    X: sp.csr_matrix
    y: np.ndarray
    w: np.ndarray
    groups: np.ndarray
    spec: DesignSpec
    game_box: pd.DataFrame     # game_idx, ps_idx, phase, <features>
    game_poss: pd.DataFrame    # game_idx, ps_idx, poss_off, poss_def
    rows: pd.DataFrame         # per-row metadata: game_idx, season, phase, poss, is_home_off, is_gt, half
    games: pd.DataFrame        # game_idx, game_id, season, phase, series_id, half

    @property
    def game_half(self) -> np.ndarray:
        """Half label ("A"/"B"/"PO") indexed by game_idx."""
        gh = np.empty(int(self.games["game_idx"].max()) + 1, dtype=object)
        gh[self.games["game_idx"].to_numpy()] = self.games["half"].to_numpy()
        return gh

    def rs_mask(self) -> np.ndarray:
        return (self.rows["phase"].to_numpy() == "RS")

    def half_mask(self, half: str, include_po: bool = False) -> np.ndarray:
        """Rows of one cross-fitting half: RS games of that half, plus (if include_po) that half's PO games.

        PO games alternate A/B within each series, so each half-fit carries half the playoff rows.
        """
        h = self.rows["half"].to_numpy()
        m = h == half
        if not include_po:
            m = m & self.rs_mask()
        return m

    def subset(self, mask: np.ndarray) -> "WindowData":
        idx = np.flatnonzero(mask)
        return WindowData(self.X[idx], self.y[idx], self.w[idx], self.groups[idx], self.spec,
                          self.game_box, self.game_poss, self.rows.iloc[idx].reset_index(drop=True), self.games)


def _order_games(st: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "season", "phase", "series_id"]
    if "game_date" in st.columns:
        cols.append("game_date")
    games = st[cols].drop_duplicates("game_id").copy()
    games["_ph"] = (games["phase"] == "PO").astype(int)
    sort_cols = ["season", "_ph"] + (["game_date"] if "game_date" in games.columns else []) + ["game_id"]
    games = games.sort_values(sort_cols).drop(columns="_ph").reset_index(drop=True)
    games["game_idx"] = np.arange(len(games), dtype=np.int64)
    # cross-fitting halves: RS games alternate A/B in chronological order within each season
    # (a whole game in one half, both teams); playoff games alternate A/B within each series.
    rs = (games["phase"] == "RS").to_numpy()
    half = np.empty(len(games), dtype=object)
    rank_rs = games[rs].groupby("season").cumcount().to_numpy()
    half[rs] = np.where(rank_rs % 2 == 0, "A", "B")
    if (~rs).any():
        rank_po = games[~rs].groupby(["season", "series_id"]).cumcount().to_numpy()
        half[~rs] = np.where(rank_po % 2 == 0, "A", "B")
    games["half"] = half
    return games


def build_game_poss(st: pd.DataFrame, ps_index: pd.Index, games: pd.DataFrame) -> pd.DataFrame:
    """Per (game, player-season) offensive and defensive possessions from the stints."""
    st = st.merge(games[["game_id", "game_idx"]], on="game_id", how="left")
    parts = []
    for slots, own, opp in ((HOME_SLOTS, "poss_h", "poss_a"), (AWAY_SLOTS, "poss_a", "poss_h")):
        for c in slots:
            parts.append(pd.DataFrame({
                "game_idx": st["game_idx"].to_numpy(),
                "ps_idx": ps_index.get_indexer(ps_key(st[c].to_numpy(), st["season"].to_numpy())),
                "poss_off": st[own].to_numpy(), "poss_def": st[opp].to_numpy()}))
    gp = pd.concat(parts, ignore_index=True)
    assert (gp["ps_idx"] >= 0).all()
    gp = gp.groupby(["game_idx", "ps_idx"], as_index=False)[["poss_off", "poss_def"]].sum()
    return gp


def build_design(stints: pd.DataFrame, box: pd.DataFrame, features: list, cfg: dict,
                 gt_weight: float | None = None, margin_bins: bool = False) -> WindowData:
    """Build the window design from stints (one row per stint) and per-game player box counts.

    stints columns: STINT_COLUMNS (+ optional game_date).
    box columns: game_id, player_id, season, phase, <features>.
    """
    st = stints.reset_index(drop=True).copy()
    for c in HOME_SLOTS + AWAY_SLOTS:
        st[c] = st[c].astype(np.int64)
    if gt_weight is None:
        gt_weight = float(cfg.get("gt_weight", 1.0))
    margin_clip = float(cfg.get("margin_clip", 25))

    games = _order_games(st)
    st = st.merge(games[["game_id", "game_idx"]], on="game_id", how="left")

    # player-season table: everyone who appears in a stint or a box row
    keys = [ps_key(st[c].to_numpy(), st["season"].to_numpy()) for c in HOME_SLOTS + AWAY_SLOTS]
    keys.append(ps_key(box["player_id"].to_numpy(), box["season"].to_numpy()))
    keys = np.unique(np.concatenate(keys))
    ps_table = pd.DataFrame({"key": keys, "season": keys // 100_000_000, "player_id": keys % 100_000_000})
    ps_table = ps_table.sort_values(["season", "player_id"]).reset_index(drop=True)
    ps_table["ps_idx"] = np.arange(len(ps_table), dtype=np.int64)
    ps_index = pd.Index(ps_table["key"].to_numpy())
    seasons = sorted(ps_table["season"].unique().tolist())
    season_of_ps = np.searchsorted(np.asarray(seasons), ps_table["season"].to_numpy())
    n_ps = len(ps_table)

    # rows: two per stint (home offense, away offense)
    n_st = len(st)
    slot_idx = {c: ps_index.get_indexer(ps_key(st[c].to_numpy(), st["season"].to_numpy())) for c in HOME_SLOTS + AWAY_SLOTS}
    home_ps = np.stack([slot_idx[c] for c in HOME_SLOTS], axis=1)
    away_ps = np.stack([slot_idx[c] for c in AWAY_SLOTS], axis=1)
    assert (home_ps >= 0).all() and (away_ps >= 0).all()
    is_po = (st["phase"].to_numpy() == "PO")
    neutral = st["neutral"].to_numpy().astype(bool) if "neutral" in st.columns else np.zeros(n_st, bool)
    is_gt = st["is_gt"].to_numpy().astype(bool)
    margin_h = st["margin_h"].to_numpy().astype(float)
    frac_rem = st["frac_rem"].to_numpy().astype(float)
    season_idx = np.searchsorted(np.asarray(seasons), st["season"].to_numpy())

    sides = []
    for off_ps, def_ps, poss_c, pts_c, sign in ((home_ps, away_ps, "poss_h", "pts_h", 1.0), (away_ps, home_ps, "poss_a", "pts_a", -1.0)):
        poss = st[poss_c].to_numpy().astype(float)
        pts = st[pts_c].to_numpy().astype(float)
        keep = poss > 0
        sides.append(dict(off=off_ps[keep], de=def_ps[keep], poss=poss[keep], pts=pts[keep],
                          home=np.where(neutral[keep], 0.0, sign), margin=sign * margin_h[keep],
                          stint=np.flatnonzero(keep), is_home_off=sign > 0))
    off = np.concatenate([s["off"] for s in sides]); de = np.concatenate([s["de"] for s in sides])
    poss = np.concatenate([s["poss"] for s in sides]); pts = np.concatenate([s["pts"] for s in sides])
    home = np.concatenate([s["home"] for s in sides]); margin = np.concatenate([s["margin"] for s in sides])
    stint_i = np.concatenate([s["stint"] for s in sides])
    is_home_off = np.concatenate([np.full(len(s["stint"]), s["is_home_off"]) for s in sides])
    n = len(poss)

    y = 100.0 * pts / poss
    w = poss * np.where(is_gt[stint_i], gt_weight, 1.0)
    r_po = is_po[stint_i].astype(float)
    r_gt = is_gt[stint_i].astype(float)
    r_margin = np.clip(margin, -margin_clip, margin_clip)
    r_frem = frac_rem[stint_i]
    r_season = season_idx[stint_i]
    r_game = st["game_idx"].to_numpy()[stint_i]

    # F block
    f_names = ["home"] + [f"int_{s}" for s in seasons] + ["is_po", "po_home", "is_gt"]
    f_cols = [home] + [(r_season == k).astype(float) for k in range(len(seasons))] + [r_po, r_po * home, r_gt]
    if margin_bins:
        edges = np.asarray(cfg["margin_bins"], dtype=float)
        nb = len(edges) - 1
        b = np.clip(np.searchsorted(edges, r_margin, side="right") - 1, 0, nb - 1)
        halves = int(cfg.get("time_halves", 2))
        h = np.clip((halves * (1.0 - r_frem)).astype(int), 0, halves - 1)
        for hh in range(halves):
            for bb in range(nb):
                if bb == nb // 2 and hh == 0:
                    continue  # reference cell
                f_names.append(f"mbin{bb}_t{hh}")
                f_cols.append(((b == bb) & (h == hh)).astype(float))
    else:
        f_names += ["margin", "margin_frem"]
        f_cols += [r_margin, r_margin * r_frem]
    F = sp.csr_matrix(np.column_stack(f_cols))

    rows_ar = np.repeat(np.arange(n), 5)
    Z_O = sp.csr_matrix((np.ones(n * 5), (rows_ar, off.ravel())), shape=(n, n_ps))
    Z_D = sp.csr_matrix((np.ones(n * 5), (rows_ar, de.ravel())), shape=(n, n_ps))
    assert Z_O.nnz == n * 5 and Z_D.nnz == n * 5, "duplicate player ids inside a lineup"
    G = sp.csr_matrix(r_game.astype(float)[:, None])
    X = sp.hstack([Z_O, Z_D, F, G], format="csr")
    X.sort_indices()

    # groups: game for RS rows, series for PO rows
    series = st["series_id"].astype(str).to_numpy()[stint_i]
    gid = st["game_id"].astype(str).to_numpy()[stint_i]
    groups = np.where(r_po > 0, np.char.add("S:", series.astype(str)), np.char.add("G:", gid.astype(str)))

    # side tables
    game_poss = build_game_poss(st.drop(columns="game_idx"), ps_index, games)
    bx = box.merge(games[["game_id", "game_idx"]], on="game_id", how="inner")
    bx["ps_idx"] = ps_index.get_indexer(ps_key(bx["player_id"].to_numpy(), bx["season"].to_numpy()))
    game_box = bx[["game_idx", "ps_idx", "phase"] + list(features)].copy()
    game_box = game_box.groupby(["game_idx", "ps_idx", "phase"], as_index=False)[list(features)].sum()

    # column groups for lam_buckets
    rs_games = games.loc[games["phase"] == "RS", "game_idx"].to_numpy()
    gp_rs = game_poss[np.isin(game_poss["game_idx"], rs_games)]
    ps_poss = np.zeros(n_ps)
    np.add.at(ps_poss, gp_rs["ps_idx"].to_numpy(), gp_rs["poss_off"].to_numpy())
    thr = float(cfg.get("low_poss_threshold", 500))
    thr_hi = float(cfg.get("starter_poss_threshold", 1500))
    low = np.flatnonzero(ps_poss < thr)
    high = np.flatnonzero(ps_poss >= thr_hi)
    col_groups = {"low_poss": np.concatenate([low, low + n_ps]),
                  "high_poss": np.concatenate([high, high + n_ps]),
                  "high_poss_O": high, "high_poss_D": high + n_ps}

    spec = DesignSpec(n_ps=n_ps, seasons=seasons, season_of_ps=season_of_ps,
                      ps_table=ps_table[["ps_idx", "player_id", "season"]].copy(),
                      f_names=f_names, features=list(features), col_groups=col_groups)
    game_half = np.empty(len(games), dtype=object)
    game_half[games["game_idx"].to_numpy()] = games["half"].to_numpy()
    rows = pd.DataFrame({"game_idx": r_game, "season": np.asarray(seasons)[r_season], "phase": np.where(r_po > 0, "PO", "RS"),
                         "poss": poss, "is_home_off": is_home_off, "is_gt": r_gt > 0, "stint": stint_i,
                         "half": game_half[r_game]})
    return WindowData(X=X, y=y, w=w, groups=groups, spec=spec, game_box=game_box, game_poss=game_poss, rows=rows, games=games)
