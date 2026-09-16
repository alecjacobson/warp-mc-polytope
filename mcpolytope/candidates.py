import numpy as np
import warp as wp


def alias_table(weights):
    """Build a Walker's alias table for O(1) weighted categorical sampling.

    Returns (prob, alias): float32[K] and int32[K] such that, for a slot i
    drawn uniformly from [0,K), sample_alias accepts i with probability
    prob[i] and otherwise redirects to alias[i] -- both branches land on
    outcomes distributed according to `weights`.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] == 0:
        raise ValueError("weights must be a nonempty 1D array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must not be all-zero")

    K = weights.shape[0]
    scaled = weights * (K / total)  # mean 1.0

    prob = np.zeros(K, dtype=np.float64)
    alias = np.zeros(K, dtype=np.int64)

    small = [i for i in range(K) if scaled[i] < 1.0]
    large = [i for i in range(K) if scaled[i] >= 1.0]

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1.0 - scaled[s])
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    # leftover entries are numerically ~1.0 due to floating point error
    for i in large:
        prob[i] = 1.0
    for i in small:
        prob[i] = 1.0

    return prob.astype(np.float32), alias.astype(np.int32)


@wp.func
def sample_alias(state: wp.uint32, prob: wp.array(dtype=wp.float32), alias: wp.array(dtype=wp.int32), K: int):
    """Returns (new_state, index). A wp.uint32 RNG state passed into a
    wp.func is NOT mutated-in-place from the caller's perspective (that only
    happens for consecutive builtin calls within the *same* function body) --
    callers MUST reassign their local `state` from the returned value, or
    every call in a loop will silently draw the exact same index. (This was
    a real bug: see git history -- with replacement-sampled candidate
    directions were previously identical across an entire trial.)
    """
    u = wp.randf(state)
    i = wp.int32(u * float(K))
    if i >= K:
        i = K - 1
    coin = wp.randf(state)
    if coin < prob[i]:
        return state, i
    return state, alias[i]
