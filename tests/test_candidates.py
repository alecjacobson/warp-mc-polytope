import numpy as np
import pytest
import warp as wp

from mcpolytope.candidates import alias_table, sample_alias, sample_vmf

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


@wp.kernel(enable_backward=False)
def _vmf_kernel(seed: int, mu: wp.vec3, kappa: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    _state, d = sample_vmf(state, mu, kappa)
    out[tid] = d


def draw_vmf(mu, kappa, num_draws, seed=0, device="cpu"):
    out = wp.zeros(num_draws, dtype=wp.vec3, device=device)
    wp.launch(_vmf_kernel, dim=num_draws, inputs=[seed, wp.vec3(*mu), kappa, out], device=device)
    return out.numpy()


def _mean_resultant_length_theory(kappa):
    # A(kappa) = coth(kappa) - 1/kappa, the S^2 vMF mean resultant length
    return 1.0 / np.tanh(kappa) - 1.0 / kappa


@pytest.mark.parametrize("kappa", [1.0, 10.0, 100.0, 1000.0])
def test_vmf_mean_resultant_length_matches_theory(kappa):
    mu = np.array([0.0, 0.0, 1.0])
    d = draw_vmf(mu, kappa, 300_000)
    rbar = np.linalg.norm(d.mean(axis=0))
    assert rbar == pytest.approx(_mean_resultant_length_theory(kappa), abs=0.01)


def test_vmf_kappa_zero_is_uniform_over_sphere():
    mu = np.array([0.0, 0.0, 1.0])
    d = draw_vmf(mu, 0.0, 300_000)
    # mean should be ~0 (isotropic) and per-axis variance ~1/3
    assert np.linalg.norm(d.mean(axis=0)) < 0.01
    np.testing.assert_allclose(d.var(axis=0), np.full(3, 1.0 / 3.0), atol=0.01)


def test_vmf_large_kappa_concentrates_tightly():
    mu = np.array([0.3, -0.5, 0.81])
    mu = mu / np.linalg.norm(mu)
    d = draw_vmf(mu, 1.0e5, 50_000)
    dots = d @ mu
    assert dots.min() > 0.99  # within ~8 degrees of mu for every single draw


def test_vmf_samples_are_unit_length():
    mu = np.array([1.0, 0.0, 0.0])
    for kappa in [0.0, 5.0, 500.0]:
        d = draw_vmf(mu, kappa, 5000)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)


def test_vmf_gpu_matches_cpu():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    mu = np.array([0.2, 0.6, -0.77])
    mu = mu / np.linalg.norm(mu)
    d_cpu = draw_vmf(mu, 50.0, 200_000, seed=3, device="cpu")
    d_gpu = draw_vmf(mu, 50.0, 200_000, seed=3, device="cuda:0")
    rbar_cpu = np.linalg.norm(d_cpu.mean(axis=0))
    rbar_gpu = np.linalg.norm(d_gpu.mean(axis=0))
    assert rbar_gpu == pytest.approx(rbar_cpu, abs=0.01)


def test_gpu_matches_cpu_distribution():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    weights = np.array([1.0, 5.0, 2.0, 8.0])
    draws_cpu = draw_many(weights, 200_000, seed=1, device="cpu")
    draws_gpu = draw_many(weights, 200_000, seed=1, device="cuda:0")
    freq_cpu = np.bincount(draws_cpu, minlength=4) / len(draws_cpu)
    freq_gpu = np.bincount(draws_gpu, minlength=4) / len(draws_gpu)
    np.testing.assert_allclose(freq_gpu, freq_cpu, atol=0.02)
