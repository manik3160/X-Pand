"""
src/bip_optimizer.py
=====================
Binary Integer Programming (BIP) optimizer for hub placement.

Uses PuLP with the CBC solver to select the optimal set of hub locations
that maximises expected profitability subject to budget (max hubs),
minimum separation distance, and probability threshold constraints.

Hard 60-second solve time limit.
"""

import math
import numpy as np
import pulp


def _haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute the Haversine great-circle distance between two points
    on Earth in kilometres.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates in decimal degrees.

    Returns
    -------
    float
        Distance in km.
    """
    R = 6371.0  # Earth radius in km
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))
    return R * c


def run_bip(gdf, prob_col, max_hubs, min_separation_km, min_prob_threshold):
    """
    Solve the Binary Integer Programming hub-placement problem.

    Maximise the sum of profitability probabilities for selected hub cells
    subject to:
        1. At most `max_hubs` cells may be selected.
        2. Any two selected cells must be at least `min_separation_km` apart.
        3. Only cells with probability ≥ `min_prob_threshold` are eligible.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Must contain `grid_id`, `centroid_lat`, `centroid_lon`, and the
        probability column named by `prob_col`.
    prob_col : str
        Name of the column holding the profitability probability (e.g.
        ``'p_profit'``).
    max_hubs : int
        Maximum number of hubs to open.
    min_separation_km : float
        Minimum Haversine distance (km) between any two selected hubs.
    min_prob_threshold : float
        Minimum probability required for a cell to be eligible.

    Returns
    -------
    selected_ids : list of str
        ``grid_id`` values of the selected hub locations.
    objective_value : float
        Total objective score (sum of probabilities of selected cells).
    """
    try:
        # ── Validate inputs ───────────────────────────────────────────
        for col in ["grid_id", "centroid_lat", "centroid_lon", prob_col]:
            if col not in gdf.columns:
                raise ValueError(
                    f"GeoDataFrame is missing required column '{col}'. "
                    f"Available: {list(gdf.columns)}"
                )

        if max_hubs < 1:
            raise ValueError(f"max_hubs must be >= 1, got {max_hubs}")
        if min_separation_km < 0:
            raise ValueError(
                f"min_separation_km must be >= 0, got {min_separation_km}"
            )

        # ── Step 1: Filter eligible cells ─────────────────────────────
        eligible = gdf[gdf[prob_col] >= min_prob_threshold].copy()
        eligible = eligible.reset_index(drop=True)
        n_eligible = len(eligible)

        if n_eligible == 0:
            print(
                f"[bip_optimizer] No cells meet the threshold "
                f"({min_prob_threshold}). Returning empty selection."
            )
            return [], 0.0

        print(
            f"[bip_optimizer] Eligible cells: {n_eligible} / {len(gdf)} "
            f"(threshold >= {min_prob_threshold})"
        )

        grid_ids = eligible["grid_id"].values.tolist()
        probs = eligible[prob_col].values.astype(float)
        lats = eligible["centroid_lat"].values.astype(float)
        lons = eligible["centroid_lon"].values.astype(float)

        # ── Step 2: Pairwise Haversine distances ─────────────────────
        conflict_pairs = []
        for i in range(n_eligible):
            for j in range(i + 1, n_eligible):
                dist = _haversine_km(lats[i], lons[i], lats[j], lons[j])
                if dist < min_separation_km:
                    conflict_pairs.append((i, j))

        print(
            f"[bip_optimizer] Separation conflicts (< {min_separation_km} km): "
            f"{len(conflict_pairs)} pairs"
        )

        # ── Step 3: Define PuLP problem ───────────────────────────────
        prob = pulp.LpProblem("hub_selection", pulp.LpMaximize)

        # ── Step 4: Decision variables ────────────────────────────────
        x = {
            grid_ids[i]: pulp.LpVariable(f"x_{grid_ids[i]}", cat="Binary")
            for i in range(n_eligible)
        }

        # ── Step 5: Objective — maximise total probability ────────────
        prob += (
            pulp.lpSum(probs[i] * x[grid_ids[i]] for i in range(n_eligible)),
            "total_profitability",
        )

        # ── Step 6: Constraints ───────────────────────────────────────
        # Budget constraint
        prob += (
            pulp.lpSum(x[gid] for gid in grid_ids) <= max_hubs,
            "max_hubs_limit",
        )

        # Separation constraints
        for pair_idx, (i, j) in enumerate(conflict_pairs):
            prob += (
                x[grid_ids[i]] + x[grid_ids[j]] <= 1,
                f"separation_{pair_idx}_{grid_ids[i]}_{grid_ids[j]}",
            )

        # ── Step 7: Solve with 60-second time limit ──────────────────
        solver = pulp.PULP_CBC_CMD(timeLimit=60, msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]

        # ── Step 8: Handle non-optimal status ─────────────────────────
        if status != "Optimal":
            print(
                f"[bip_optimizer] WARNING: BIP did not reach optimality — "
                f"status = '{status}'. Returning best feasible solution."
            )

        # ── Step 9: Extract selected hubs ─────────────────────────────
        selected_ids = [
            gid for gid in grid_ids if x[gid].varValue is not None and x[gid].varValue > 0.5
        ]

        objective_value = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else 0.0

        print(
            f"[bip_optimizer] Solution status: {status} | "
            f"Selected hubs: {len(selected_ids)} / {max_hubs} max | "
            f"Objective value: {objective_value:.4f}"
        )

        return selected_ids, objective_value

    except Exception as exc:
        raise RuntimeError(
            f"[bip_optimizer.run_bip] Failed: {exc}"
        ) from exc
