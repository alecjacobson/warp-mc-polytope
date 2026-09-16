import time

import numpy as np
import pytest
import warp as wp

from mcpolytope.search import Search


def test_gpu_throughput_smoke():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 0.5 + 0.5 * rng.random((500, 1))

    batch_size = 1 << 20
    s = Search(pts, n=20, seed=0, batch_size=batch_size, device="cuda:0")
    s.run(batch_size)  # warmup / compile
    wp.synchronize_device("cuda:0")

    num_trials = batch_size * 8
    t0 = time.perf_counter()
    s.run(num_trials)
    wp.synchronize_device("cuda:0")
    dt = time.perf_counter() - t0

    trials_per_sec = num_trials / dt
    print(f"\n{trials_per_sec:,.0f} trials/sec at n=20 on cuda:0")
    # Conservative lower bound (measured ~9.4M/sec on an L40); this is a
    # regression guard, not a target -- it should never come close to
    # failing on any GPU capable of running this at all.
    assert trials_per_sec > 1_000_000


def test_gpu_throughput_smoke_candidates():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 0.5 + 0.5 * rng.random((500, 1))
    cands = rng.standard_normal((2000, 3))
    cands = cands / np.linalg.norm(cands, axis=1, keepdims=True)
    weights = rng.random(2000) + 0.1

    batch_size = 1 << 20
    s = Search(
        pts, n=20, seed=0, batch_size=batch_size, candidates=cands, weights=weights,
        avoid_duplicates=True, device="cuda:0",
    )
    s.run(batch_size)  # warmup / compile
    wp.synchronize_device("cuda:0")

    num_trials = batch_size * 8
    t0 = time.perf_counter()
    s.run(num_trials)
    wp.synchronize_device("cuda:0")
    dt = time.perf_counter() - t0

    trials_per_sec = num_trials / dt
    print(f"\n{trials_per_sec:,.0f} trials/sec at n=20, K=2000, avoid_duplicates=True on cuda:0")
    assert trials_per_sec > 500_000


def test_gpu_throughput_smoke_kappa():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")

    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 0.5 + 0.5 * rng.random((500, 1))
    cands = rng.standard_normal((2000, 3))
    cands = cands / np.linalg.norm(cands, axis=1, keepdims=True)
    weights = rng.random(2000) + 0.1

    batch_size = 1 << 20
    s = Search(
        pts, n=20, seed=0, batch_size=batch_size, candidates=cands, weights=weights,
        kappa=50.0, device="cuda:0",
    )
    s.run(batch_size)  # warmup / compile
    wp.synchronize_device("cuda:0")

    num_trials = batch_size * 8
    t0 = time.perf_counter()
    s.run(num_trials)
    wp.synchronize_device("cuda:0")
    dt = time.perf_counter() - t0

    trials_per_sec = num_trials / dt
    print(f"\n{trials_per_sec:,.0f} trials/sec at n=20, K=2000, kappa=50 on cuda:0")
    assert trials_per_sec > 500_000
