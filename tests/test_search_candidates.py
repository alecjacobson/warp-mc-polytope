import numpy as np
import pytest
import warp as wp

from mcpolytope.search import Search


def _point_cloud(rng, m=300):
    pts = rng.standard_normal((m, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 0.5 + 0.5 * rng.random((m, 1))
    return pts


def _candidate_dirs(rng, K):
    d = rng.standard_normal((K, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_directions_come_from_candidate_set():
    rng = np.random.default_rng(0)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 64)

    s = Search(pts, n=8, seed=0, batch_size=512, candidates=cands, device="cpu")
    res = s.run(2048)

    # every returned direction must match (up to fp32 roundtrip) a candidate row
    dists = np.linalg.norm(cands[None, :, :] - res.directions[:, None, :], axis=2)
    closest = dists.min(axis=1)
    assert np.all(closest < 1e-4)


def test_weighted_candidates_valid_enclosure():
    rng = np.random.default_rng(1)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 40)
    weights = rng.random(40) + 0.1

    s = Search(pts, n=6, seed=1, batch_size=256, candidates=cands, weights=weights, device="cpu")
    res = s.run(1024)

    violation = res.directions @ pts.T - res.offsets[:, None]
    assert violation.max() < 1e-3


def test_avoid_duplicates_exact_permutation_when_k_equals_n():
    rng = np.random.default_rng(2)
    pts = _point_cloud(rng)
    n = 8
    cands = _candidate_dirs(rng, n)  # K == n

    s = Search(pts, n=n, seed=2, batch_size=64, candidates=cands, avoid_duplicates=True, device="cpu")
    s.run(64 * 5)

    # spot-check several regenerated trials (not just the winner): every
    # trial's chosen directions must be a full permutation of the pool.
    for gidx in [0, 5, 17, 42, 100, 250]:
        dirs, _offs, _vol = s.regenerate(gidx)
        dists = np.linalg.norm(cands[None, :, :] - dirs[:, None, :], axis=2)
        assigned = dists.argmin(axis=1)
        assert sorted(assigned.tolist()) == list(range(n)), f"trial {gidx} had a duplicate: {assigned}"


def test_avoid_duplicates_requires_n_le_k():
    rng = np.random.default_rng(3)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 4)
    with pytest.raises(ValueError):
        Search(pts, n=8, candidates=cands, avoid_duplicates=True, device="cpu")


def test_weights_requires_candidates():
    rng = np.random.default_rng(4)
    pts = _point_cloud(rng)
    with pytest.raises(ValueError):
        Search(pts, n=6, weights=np.ones(10), device="cpu")


def test_regenerate_matches_recorded_best_candidates():
    rng = np.random.default_rng(5)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 50)
    s = Search(pts, n=10, seed=5, batch_size=1024, candidates=cands, device="cpu")
    res = s.run(4096)
    _dirs, _offs, vol = s.regenerate(res.index)
    assert vol == pytest.approx(float(s.global_best_vol.numpy()[0]), rel=1e-5)


def test_gpu_matches_cpu_candidates():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(6)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 60)
    weights = rng.random(60) + 0.1

    s_cpu = Search(pts, n=8, seed=6, batch_size=1024, candidates=cands, weights=weights, device="cpu")
    res_cpu = s_cpu.run(4096)
    s_gpu = Search(pts, n=8, seed=6, batch_size=1024, candidates=cands, weights=weights, device="cuda:0")
    res_gpu = s_gpu.run(4096)

    assert res_gpu.index == res_cpu.index
    assert res_gpu.volume == pytest.approx(res_cpu.volume, rel=1e-4)
    np.testing.assert_allclose(res_gpu.directions, res_cpu.directions, atol=1e-5)


def test_graph_capture_matches_ungraphed_candidates():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(7)
    pts = _point_cloud(rng)
    cands = _candidate_dirs(rng, 40)

    s1 = Search(pts, n=8, seed=7, batch_size=1024, candidates=cands, avoid_duplicates=True, device="cuda:0")
    res1 = s1.run(1024 * 8)
    s2 = Search(pts, n=8, seed=7, batch_size=1024, candidates=cands, avoid_duplicates=True, device="cuda:0")
    res2 = s2.run(1024 * 8, use_graph=True)

    assert res2.index == res1.index
    assert res2.volume == pytest.approx(res1.volume, rel=1e-4)
