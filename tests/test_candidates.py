import numpy as np
import pytest
import warp as wp

from mcpolytope.candidates import alias_table, sample_alias

wp.init()


@wp.kernel(enable_backward=False)
def _sample_kernel(
    seed: int,
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    _state, idx = sample_alias(state, prob, alias, K)
    out[tid] = idx


def draw_many(weights, num_draws, seed=0, device="cpu"):
    prob_np, alias_np = alias_table(weights)
    prob = wp.array(prob_np, dtype=wp.float32, device=device)
    alias = wp.array(alias_np, dtype=wp.int32, device=device)
    out = wp.zeros(num_draws, dtype=wp.int32, device=device)
    wp.launch(_sample_kernel, dim=num_draws, inputs=[seed, prob, alias, len(weights), out], device=device)
    return out.numpy()


def test_matches_weights_statistically():
    weights = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    draws = draw_many(weights, 400_000)
    counts = np.bincount(draws, minlength=len(weights))
    freq = counts / counts.sum()
    expected = weights / weights.sum()
    np.testing.assert_allclose(freq, expected, atol=0.01)


def test_zero_weight_category_never_drawn():
    weights = np.array([1.0, 0.0, 5.0, 0.0, 2.0])
    draws = draw_many(weights, 200_000)
    counts = np.bincount(draws, minlength=len(weights))
    assert counts[1] == 0
    assert counts[3] == 0
    assert counts[0] > 0 and counts[2] > 0 and counts[4] > 0


def test_single_category():
    draws = draw_many(np.array([5.0]), 1000)
    assert np.all(draws == 0)


def test_uniform_weights_match_uniform_draw():
    weights = np.ones(8)
    draws = draw_many(weights, 400_000)
    counts = np.bincount(draws, minlength=8)
    freq = counts / counts.sum()
    np.testing.assert_allclose(freq, np.full(8, 1.0 / 8), atol=0.01)


@pytest.mark.parametrize(
    "bad_weights",
    [
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 2.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([]),
    ],
)
def test_rejects_invalid_weights(bad_weights):
    with pytest.raises(ValueError):
        alias_table(bad_weights)


@wp.kernel(enable_backward=False)
def _sample_n_kernel(
    seed: int,
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    n: int,
    out: wp.array(dtype=wp.int32, ndim=2),
):
    """Draws n samples per thread, reusing/advancing one RNG state across
    all n calls -- the pattern _sample_trial_candidates uses. Regression
    test for a real bug: sample_alias's wp.uint32 state parameter is NOT
    mutated-in-place across a wp.func call boundary, so failing to
    reassign `state` from its return value made every draw within a trial
    identical.
    """
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    for k in range(n):
        state, idx = sample_alias(state, prob, alias, K)
        out[tid, k] = idx


def test_successive_draws_within_one_thread_are_not_all_identical():
    weights = np.ones(50)
    prob_np, alias_np = alias_table(weights)
    prob = wp.array(prob_np, dtype=wp.float32, device="cpu")
    alias = wp.array(alias_np, dtype=wp.int32, device="cpu")
    n = 16
    num_threads = 64
    out = wp.zeros((num_threads, n), dtype=wp.int32, device="cpu")
    wp.launch(_sample_n_kernel, dim=num_threads, inputs=[0, prob, alias, 50, n, out], device="cpu")
    draws = out.numpy()
    # every thread's n draws must not all collapse to a single repeated value
    for row in draws:
        assert len(set(row.tolist())) > 1, f"all draws identical: {row}"


def test_gpu_matches_cpu_distribution():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    weights = np.array([1.0, 5.0, 2.0, 8.0])
    draws_cpu = draw_many(weights, 200_000, seed=1, device="cpu")
    draws_gpu = draw_many(weights, 200_000, seed=1, device="cuda:0")
    freq_cpu = np.bincount(draws_cpu, minlength=4) / len(draws_cpu)
    freq_gpu = np.bincount(draws_gpu, minlength=4) / len(draws_gpu)
    np.testing.assert_allclose(freq_gpu, freq_cpu, atol=0.02)
