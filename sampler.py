import random
import numpy as np
from itertools import product
from scipy.stats import qmc

# Helper functions to flatten and rebuild the design variable dictionary.
# The design variable ranges (dv_ranges) must be provided by the caller.
# Format reminder:
#   Scalars: key -> (min, max)
#   Vectors: key -> [(min, max), ...]
def _flatten_dv_ranges(dv_ranges):
    """
    Convert dv_ranges dict into a flat list of (key, subindex, lower, upper) tuples.
    For scalars, subindex is None.
    """
    flat = []
    for key, rng in dv_ranges.items():
        if isinstance(rng, tuple) and all(isinstance(x, (int, float)) for x in rng):
            flat.append((key, None, rng[0], rng[1]))
        elif isinstance(rng, (list, tuple)):
            # assume a list of tuples for vector variables
            for i, sub_rng in enumerate(rng):
                flat.append((key, i, sub_rng[0], sub_rng[1]))
        else:
            raise ValueError(f"Unsupported range format for key {key}")
    return flat

def _unflatten_sample(flat_sample, dv_ranges):
    """
    Given a flat sample (list or array of floats) and the original dv_ranges,
    reconstruct a dictionary with the same structure as dv_ranges.
    """
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    sample = {}
    for i, (key, subindex, lower, upper) in enumerate(flat_ranges):
        if subindex is None:
            sample[key] = flat_sample[i]
        else:
            if key not in sample:
                sample[key] = []
            sample[key].append(flat_sample[i])
    return sample

# Random Sampling (Monte Carlo)
def sample_random(n, dv_ranges):
    """
    Generate n samples using pure random sampling.
    Returns a list of dictionaries.
    """
    samples = []
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    for _ in range(n):
        flat_sample = []
        for (key, subindex, lower, upper) in flat_ranges:
            flat_sample.append(random.uniform(lower, upper))
        samples.append(_unflatten_sample(flat_sample, dv_ranges))
    return samples

# Latin Hypercube Sampling (LHS)
def sample_lhs(n, dv_ranges):
    """
    Generate n samples using Latin Hypercube Sampling.
    Returns a list of dictionaries.
    """
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    d = len(flat_ranges)
    sampler = qmc.LatinHypercube(d=d)
    raw_samples = sampler.random(n=n)
    l_bounds = [fr[2] for fr in flat_ranges]
    u_bounds = [fr[3] for fr in flat_ranges]
    scaled_samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    return [_unflatten_sample(s, dv_ranges) for s in scaled_samples]

# Sobol Sampling (Quasi-Random Low-discrepancy Sequence)
def sample_sobol(n, dv_ranges):
    """
    Generate n samples using a Sobol sequence.
    Note: n should ideally be a power of 2 for Sobol.
    Returns a list of dictionaries.
    """
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    d = len(flat_ranges)
    sampler = qmc.Sobol(d=d, scramble=True)
    raw_samples = sampler.random(n=n)
    l_bounds = [fr[2] for fr in flat_ranges]
    u_bounds = [fr[3] for fr in flat_ranges]
    scaled_samples = qmc.scale(raw_samples, l_bounds, u_bounds)
    return [_unflatten_sample(s, dv_ranges) for s in scaled_samples]

# Grid Sampling
def sample_grid(n_points_per_dim, dv_ranges):
    """
    Generate samples by creating a grid.
    n_points_per_dim: number of grid points for each flat dimension.
    Returns a list of dictionaries.
    Warning: Total samples = n_points_per_dim^(number of flat dimensions).
    """
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    grids = [np.linspace(lower, upper, n_points_per_dim) for (_, _, lower, upper) in flat_ranges]
    grid_points = list(product(*grids))
    return [_unflatten_sample(np.array(point), dv_ranges) for point in grid_points]

# Stratified Sampling
def sample_stratified(n, dv_ranges):
    """
    Generate n samples using stratified sampling.
    Divides each flat dimension into n strata and picks a random value from each stratum.
    Returns a list of dictionaries.
    """
    flat_ranges = _flatten_dv_ranges(dv_ranges)
    d = len(flat_ranges)
    strata = []
    for key, subindex, lower, upper in flat_ranges:
        edges = np.linspace(lower, upper, n + 1)
        strata.append([random.uniform(edges[i], edges[i + 1]) for i in range(n)])
    for dim in range(d):
        np.random.shuffle(strata[dim])
    samples = []
    for i in range(n):
        flat_sample = [strata[dim][i] for dim in range(d)]
        samples.append(_unflatten_sample(flat_sample, dv_ranges))
    return samples

# Orthogonal Sampling
def sample_orthogonal(n, dv_ranges):
    """
    Generate n samples using an orthogonal sampling design.
    Here we approximate orthogonal sampling using a Latin Hypercube with scramble.
    Returns a list of dictionaries.
    """
    return sample_lhs(n, dv_ranges)

# Example: Function to get samples by a chosen method.
def get_samples(method="random", n=10, n_points_per_dim=5, dv_ranges=None):
    """
    Retrieve samples based on the selected method.
    The dv_ranges must be provided by the caller.
    method: one of "random", "lhs", "sobol", "grid", "stratified", "orthogonal"
    """
    if dv_ranges is None:
        raise ValueError("Design variable ranges (dv_ranges) must be provided by the caller.")
    if method == "random":
        return sample_random(n, dv_ranges)
    elif method == "lhs":
        return sample_lhs(n, dv_ranges)
    elif method == "sobol":
        return sample_sobol(n, dv_ranges)
    elif method == "grid":
        return sample_grid(n_points_per_dim, dv_ranges)
    elif method == "stratified":
        return sample_stratified(n, dv_ranges)
    elif method == "orthogonal":
        return sample_orthogonal(n, dv_ranges)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

# For testing purposes, you can run this module directly.
# In actual use, generator.py should define the DV_RANGES and pass them here.
if __name__ == "__main__":
    # Example DV_RANGES for testing.
    test_dv_ranges = {
        "tw_slat": (-10, 10),
        "tw_flap": (-10, 10),
        "of_slat": [(0.005, 0.03), (-0.02, 0.02)],
        "of_flap": [(0.005, 0.03), (-0.02, 0.02)]
    }
    method = "lhs"
    n_samples = 10
    samples = get_samples(method, n_samples, dv_ranges=test_dv_ranges)
    for i, s in enumerate(samples):
        print(f"Sample {i}: {s}")