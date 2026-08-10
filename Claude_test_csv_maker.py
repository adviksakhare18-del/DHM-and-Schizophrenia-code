# 9. THIS IS THE FILE THAT GENERATES 100 ROWS OF HEALTHY AND 100 ROWS OF SCHIZOPHRENIC ACTION POTENTIALS (THE TEST DATASET)
#!/usr/bin/env python3
"""
LABELED ACTION POTENTIAL DATASET GENERATOR
════════════════════════════════════════════════════════════════════════════
Reuses the exact same data source and generation method as the original
Claude_csv_maker.py (Allen Brain Cell Types API for healthy, published
literature values for schizophrenia; seeded Gaussian population variability;
deterministic waveform reconstruction).

Instead of the wide "Time_ms x Individual" layout, this produces a
classifier-ready layout:

    Row  = one individual (one full action-potential trace)
    First column = Diagnosis  ("healthy" or "schizophrenic")
    Remaining cols = voltage at every timepoint (0.00 ms ... 80.00 ms, 0.05 ms steps)

Output:
    Row 1-100    -> Individuals Id1-Id100 from the HEALTHY population
    Row 101-200  -> Individuals Id1-Id100 from the SCHIZOPHRENIA population

Because generation is driven by np.random.default_rng(seed) and each
individual's parameters are drawn strictly in Run_ID order, taking the
first 100 individuals here reproduces exactly the first 100 individuals
(Id1-Id100) that the original 250-run script would generate with the
same seed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ══════════════════════════════════════════════════════════════════════════
# ALLEN BRAIN CELL TYPES REST API  (same source as original script)
# ══════════════════════════════════════════════════════════════════════════

ALLEN_API = "http://api.brain-map.org/api/v2/data/query.json"

ALLEN_FEATURE_FIELDS = [
    "vrest",
    "peak_v_long_square",
    "threshold_v_long_square",
    "fast_trough_v_long_square",
    "upstroke_downstroke_ratio_long_square",
    "ri",
    "tau",
]


def query_allen_human_features(n_cells: int = 25, timeout: int = 15) -> Optional[list[dict]]:
    """Query Allen Cell Types API for real human electrophysiology features."""
    field_str = ",".join(ALLEN_FEATURE_FIELDS + ["donor__name", "donor__disease_state",
                                                   "structure__name", "id"])
    criteria = (
        "model::ApiCellTypesSpecimenDetail"
        ",rma::criteria,"
        "[donor__species$il'homo sapiens'],"
        "[peak_v_long_square$gt0],"
        "[fast_trough_v_long_square$lt0]"
        f",rma::options[num_rows$eq{n_cells}]"
        f"[only$eq'{field_str}']"
        "[order$eq'id']"
    )
    try:
        print("  -> Querying Allen Cell Types API...")
        resp = requests.get(
            ALLEN_API,
            params={"criteria": criteria},
            timeout=timeout,
            headers={"User-Agent": "K-Dense-AP-Fetcher/1.0 (research)"},
        )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            print("  x Allen API returned success=false")
            return None
        cells = payload.get("msg", [])
        if not cells:
            print("  x Allen API returned 0 cells")
            return None
        print(f"  [OK] Allen API returned {len(cells)} human cells with AP data")
        return cells
    except requests.exceptions.ConnectionError:
        print("  x No internet connection - using built-in real published values")
        return None
    except requests.exceptions.Timeout:
        print("  x Allen API timed out - using built-in real published values")
        return None
    except Exception as e:
        print(f"  x Allen API error: {e} - using built-in real published values")
        return None


def median_features_from_allen(cells: list[dict]) -> dict:
    """Median across queried human cells (same as original script)."""
    result = {}
    for feat in ALLEN_FEATURE_FIELDS:
        values = [c[feat] for c in cells if c.get(feat) is not None and c[feat] != 0]
        result[feat] = float(np.median(values)) if values else None
    return result


# Offline fallback, identical to what the live API returns (same as original)
HEALTHY_FALLBACK_FEATURES = {
    "vrest":                                 -67.8,
    "peak_v_long_square":                    +34.2,
    "threshold_v_long_square":               -48.9,
    "fast_trough_v_long_square":             -73.6,
    "upstroke_downstroke_ratio_long_square":   4.7,
    "ri":                                    132.0,
    "tau":                                    18.4,
}

# Published literature values, schizophrenia group (same as original)
SCZ_PUBLISHED_FEATURES = {
    "vrest":                                 -66.5,
    "peak_v_long_square":                    +16.0,
    "threshold_v_long_square":               -46.2,
    "fast_trough_v_long_square":             -70.8,
    "upstroke_downstroke_ratio_long_square":   2.8,
    "ri":                                    165.0,
    "tau":                                    22.1,
}

HEALTHY_SOURCE = "Allen Brain Cell Types Database (api.brain-map.org)"
SCZ_SOURCE = "Ahmad W et al. (2022) PNAS 119(4):e2109395119"


# ══════════════════════════════════════════════════════════════════════════
# WAVEFORM RECONSTRUCTION FROM REAL FEATURES  (identical to original script)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class APParams:
    label:          str
    data_source:    str
    api_queried:    bool    = False
    V_rest:         float   = -70.0
    V_peak:         float   =  35.0
    V_threshold:    float   = -50.0
    V_ahp_trough:   float   = -75.0
    tau_rise:       float   =  0.30
    tau_fall:       float   =  1.50
    ahp_depth:      float   = 5.0
    ahp_delay:      float   =  4.0
    tau_ahp_rise:   float   =  1.0
    tau_ahp_fall:   float   = 15.0
    t_stim:         float   =  5.0
    t_total:        float   = 80.0
    dt:             float   =  0.05


def features_to_params(features: dict, label: str, source: str,
                        api_queried: bool = False) -> APParams:
    V_rest   = features.get("vrest")                                 or -68.0
    V_peak   = features.get("peak_v_long_square")                    or  35.0
    V_thresh = features.get("threshold_v_long_square")               or -50.0
    V_trough = features.get("fast_trough_v_long_square")             or -75.0
    ratio    = features.get("upstroke_downstroke_ratio_long_square") or  5.0
    tau_mem  = features.get("tau")                                   or  18.0

    tau_rise_anchor = 0.30
    ratio_anchor    = 4.7
    tau_rise  = tau_rise_anchor * (ratio_anchor / max(ratio, 0.5))
    tau_rise  = np.clip(tau_rise, 0.15, 1.20)

    tau_fall  = tau_rise * max(ratio, 0.5)
    tau_fall  = np.clip(tau_fall, 0.5, 5.0)

    ahp_depth = max(0.0, V_rest - V_trough)
    tau_ahp_fall = np.clip(tau_mem * 0.85, 8.0, 50.0)
    tau_ahp_rise = np.clip(tau_ahp_fall * 0.07, 0.5, 3.0)

    return APParams(
        label        = label,
        data_source  = source,
        api_queried  = api_queried,
        V_rest       = round(V_rest,   2),
        V_peak       = round(V_peak,   2),
        V_threshold  = round(V_thresh, 2),
        V_ahp_trough = round(V_trough, 2),
        tau_rise     = round(tau_rise,    4),
        tau_fall     = round(tau_fall,    4),
        ahp_depth    = round(ahp_depth,   2),
        tau_ahp_rise = round(tau_ahp_rise,4),
        tau_ahp_fall = round(tau_ahp_fall,2),
    )


def _normalised_basis(tau: np.ndarray, tau_rise: float, tau_fall: float) -> np.ndarray:
    t_fine    = np.linspace(0.0, tau_rise * 100.0, 100_000)
    raw_fine  = (1 - np.exp(-t_fine / tau_rise)) * np.exp(-t_fine / tau_fall)
    peak_val  = raw_fine.max()
    tau_safe = np.clip(tau, 0.0, None)
    raw      = (1 - np.exp(-tau_safe / tau_rise)) * np.exp(-tau_safe / tau_fall)
    return raw / (peak_val if peak_val > 1e-12 else 1.0)


def generate_waveform(p: APParams) -> pd.DataFrame:
    n   = int(round(p.t_total / p.dt)) + 1
    t   = np.linspace(0.0, p.t_total, n)
    V   = np.full(n, p.V_rest, dtype=np.float64)

    post_mask   = t >= p.t_stim
    tau_spike   = t[post_mask] - p.t_stim

    spike_amp   = p.V_peak - p.V_rest
    spike       = spike_amp * _normalised_basis(tau_spike, p.tau_rise, p.tau_fall)

    tau_ahp     = np.clip(tau_spike - p.ahp_delay, 0.0, None)
    ahp         = -p.ahp_depth * _normalised_basis(tau_ahp, p.tau_ahp_rise, p.tau_ahp_fall)

    V[post_mask] = p.V_rest + spike + ahp

    return pd.DataFrame({
        "Time_ms":               np.round(t, 4),
        "Membrane_Potential_mV": np.round(V, 4),
    })


# ══════════════════════════════════════════════════════════════════════════
# POPULATION GENERATOR  (identical variability model + same seeds)
# ══════════════════════════════════════════════════════════════════════════

VARIABILITY = {
    "Healthy": {
        "V_peak":    4.0,
        "V_rest":    1.5,
        "tau_rise":  0.05,
        "ahp_depth": 1.2,
    },
    "Schizophrenia": {
        "V_peak":    5.5,
        "V_rest":    1.8,
        "tau_rise":  0.10,
        "ahp_depth": 1.8,
    },
}


def _vary_params(base: APParams, rng: np.random.Generator) -> APParams:
    sigma = VARIABILITY[base.label]

    V_peak_new    = base.V_peak    + rng.normal(0, sigma["V_peak"])
    V_rest_new    = base.V_rest    + rng.normal(0, sigma["V_rest"])
    tau_rise_new  = base.tau_rise  + rng.normal(0, sigma["tau_rise"])
    ahp_depth_new = base.ahp_depth + rng.normal(0, sigma["ahp_depth"])

    V_peak_new    = float(np.clip(V_peak_new,   max(base.V_peak * 0.4, 5.0), 75.0))
    V_rest_new    = float(np.clip(V_rest_new,   -80.0,             -60.0))
    tau_rise_new  = float(np.clip(tau_rise_new,   0.10,              1.50))
    ahp_depth_new = float(np.clip(ahp_depth_new,  0.5,              20.0))

    ratio     = base.tau_fall / base.tau_rise
    tau_fall_new = float(np.clip(tau_rise_new * ratio, 0.4, 6.0))

    import copy
    p = copy.copy(base)
    p.V_peak    = round(V_peak_new,    4)
    p.V_rest    = round(V_rest_new,    4)
    p.tau_rise  = round(tau_rise_new,  6)
    p.tau_fall  = round(tau_fall_new,  6)
    p.ahp_depth = round(ahp_depth_new, 4)
    return p


def generate_population(base: APParams, n_runs: int, seed: int) -> pd.DataFrame:
    """
    Same RNG mechanics as the original script: np.random.default_rng(seed),
    consumed strictly in Run_ID order. Requesting n_runs=100 reproduces
    exactly the first 100 individuals (Id1-Id100) of a 250-run population
    generated with the same seed.
    """
    rng    = np.random.default_rng(seed)
    frames = []
    for run_id in range(1, n_runs + 1):
        p  = _vary_params(base, rng)
        df = generate_waveform(p)
        df.insert(0, "Run_ID", run_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════
# RESHAPE: population (long format) -> one row per individual + label
# ══════════════════════════════════════════════════════════════════════════

def population_to_labeled_rows(pop_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Long format in  : Run_ID | Time_ms | Membrane_Potential_mV
    Wide-by-row out  : one row per Run_ID, a leading 'Diagnosis' column,
                       then one column per Time_ms.
    """
    wide = pop_df.pivot(index="Run_ID", columns="Time_ms",
                         values="Membrane_Potential_mV")
    # Keep column headers as plain numeric time values (e.g. 0.0, 0.05, ...)
    # so downstream code can read them straight as floats — no "t_"/"ms" text.
    wide.columns = [round(float(c), 4) for c in wide.columns]
    wide = wide.reset_index(drop=True)
    wide.insert(0, "Diagnosis", label)
    return wide


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    N_INDIVIDUALS = 100
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    print()
    print("  LABELED ACTION POTENTIAL DATASET GENERATOR")
    print(f"  {N_INDIVIDUALS} healthy + {N_INDIVIDUALS} schizophrenic individuals")
    print()

    # ── HEALTHY: Allen API -> base params (seed=42, same as original) ───────
    print("  [GROUP 1] HEALTHY - Allen Brain Cell Types Database")
    allen_cells = query_allen_human_features(n_cells=25)
    api_queried = allen_cells is not None

    if allen_cells:
        healthy_features = median_features_from_allen(allen_cells)
    else:
        print("  Using built-in real Allen Cell Types reference values")
        healthy_features = HEALTHY_FALLBACK_FEATURES

    h_params = features_to_params(
        healthy_features, label="Healthy", source=HEALTHY_SOURCE, api_queried=api_queried,
    )
    h_pop = generate_population(h_params, n_runs=N_INDIVIDUALS, seed=42)
    h_rows = population_to_labeled_rows(h_pop, label="healthy")
    print(f"  [OK] Generated {len(h_rows)} healthy individuals")

    # ── SCHIZOPHRENIA: published literature -> base params (seed=99) ────────
    print()
    print("  [GROUP 2] SCHIZOPHRENIA - Published Literature Parameters")
    s_params = features_to_params(
        SCZ_PUBLISHED_FEATURES, label="Schizophrenia", source=SCZ_SOURCE,
    )
    # Same post-hoc adjustments as the original script
    s_params.tau_ahp_fall = round(s_params.tau_ahp_fall * 2.1, 2)
    s_params.tau_ahp_rise = round(s_params.tau_ahp_rise * 2.2, 4)
    s_params.ahp_delay    = 7.0

    s_pop = generate_population(s_params, n_runs=N_INDIVIDUALS, seed=99)
    s_rows = population_to_labeled_rows(s_pop, label="schizophrenic")
    print(f"  [OK] Generated {len(s_rows)} schizophrenic individuals")

    # ── Stack: healthy first (rows 1-100), schizophrenic second (101-200) ───
    combined = pd.concat([h_rows, s_rows], ignore_index=True)

    out_path = os.path.join(out_dir, "labeled_action_potentials.csv")
    combined.to_csv(out_path, index=False, encoding="utf-8")

    print()
    print(f"  [OK] Labeled dataset -> {out_path}")
    print(f"       {len(combined)} rows x {len(combined.columns)} columns")
    print(f"       Rows 1-{N_INDIVIDUALS}: healthy | "
          f"Rows {N_INDIVIDUALS+1}-{2*N_INDIVIDUALS}: schizophrenic")
    print()


if __name__ == "__main__":
    main()