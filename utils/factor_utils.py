# utils/factor_utils.py
# Degradation factor parsing and manipulation for CDD-11.
#
# Four atomic degradation types: low-light (L), haze (H), rain (R), snow (S).
# A composite degradation name is formed by joining active factors with '_',
# e.g. "low_haze_rain" denotes simultaneous low-light + haze + rain.

import torch

# ============================================================================
# Factor definitions
# ============================================================================

FACTORS = ["low", "haze", "rain", "snow"]

FACTOR2IDX = {
    "low": 0,
    "haze": 1,
    "rain": 2,
    "snow": 3,
}

IDX2FACTOR = {idx: factor for factor, idx in FACTOR2IDX.items()}


# ============================================================================
# Parsing and conversion
# ============================================================================

def parse_factors(deg_name):
    """Parse a degradation name string into a list of active factors.

    Example: "low_haze_rain" → ["low", "haze", "rain"]
    """
    if deg_name is None or deg_name == "":
        return []
    factors = deg_name.split("_")
    return [f for f in factors if f in FACTOR2IDX]


def factors_to_present(factors):
    """Convert a factor list to a 4-d binary presence vector.

    Example: ["low", "haze"] → tensor([1, 1, 0, 0])
    """
    present = torch.zeros(4, dtype=torch.float32)
    for factor in factors:
        if factor in FACTOR2IDX:
            present[FACTOR2IDX[factor]] = 1.0
    return present


def build_name(factors):
    """Reconstruct a canonical degradation name from a factor list.

    Factors are sorted by their index to ensure a unique representation.
    """
    sorted_factors = sorted(factors, key=lambda f: FACTOR2IDX.get(f, 999))
    return "_".join(sorted_factors)


def get_leave_one_out_name(deg_name):
    """Return all leave-one-out degradation names.

    Example: "low_haze_rain" → ["haze_rain", "low_rain", "low_haze"]
    """
    factors = parse_factors(deg_name)
    if len(factors) <= 1:
        return []
    loo_names = []
    for i in range(len(factors)):
        remaining = factors[:i] + factors[i+1:]
        if remaining:
            loo_names.append(build_name(remaining))
    return loo_names


def present_to_factors(present):
    """Convert a presence vector back to a factor list (threshold 0.5)."""
    factors = []
    for idx, val in enumerate(present):
        if val > 0.5:
            factor = IDX2FACTOR.get(idx)
            if factor:
                factors.append(factor)
    return factors