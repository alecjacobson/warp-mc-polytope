import numpy as np
import pytest
import warp as wp

from mcpolytope.search import Search


def _point_cloud(rng, m=300):
    pts = rng.standard_normal((m, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 0.5 + 0.5 * rng.random((m, 1))
    return pts


@pytest.mark.parametrize("device", ["cpu"])
def test_search_finds_valid_enclosing_polytope(device):
    rng = np.random.default_rng(0)
    pts = _point_cloud(rng)

    s = Search(pts, n=8, seed=0, batch_size=512, device=device)
    res = s.run(2048)

    assert res.directions.shape == (8, 3)
    assert res.offsets.shape == (8,)
    assert np.isfinite(res.volume)

    # every point must lie inside every found halfspace (up to fp tolerance)
    violation = res.directions @ pts.T - res.offsets[:, None]
    assert violation.max() < 1e-3

    # the found polytope must contain (not be smaller than) the true hull
    scipy_spatial = pytest.importorskip("scipy.spatial")
    hull = scipy_spatial.ConvexHull(pts)
    assert res.volume >= hull.volume - 1e-6


def test_more_trials_never_increases_best_volume():
    rng = np.random.default_rng(1)
    pts = _point_cloud(rng)
    s = Search(pts, n=6, seed=1, batch_size=256, device="cpu")

    s.run(256)
    v1 = s.best().volume
    s.run(256 * 8)
    v2 = s.best().volume

    assert v2 <= v1 + 1e-9


def test_regenerate_matches_recorded_best():
    rng = np.random.default_rng(2)
    pts = _point_cloud(rng)
    s = Search(pts, n=10, seed=2, batch_size=1024, device="cpu")
    res = s.run(4096)

    dirs, offs, vol = s.regenerate(res.index)
    assert vol == pytest.approx(
        # recompute the recentered-frame volume the same way best() reports it (pre centroid-shift)
        float(s.global_best_vol.numpy()[0]),
        rel=1e-5,
    )


def test_trace_is_monotonically_nonincreasing():
    rng = np.random.default_rng(3)
    pts = _point_cloud(rng)
    s = Search(pts, n=8, seed=3, batch_size=128, device="cpu")
    s.run(128 * 20, trace_every_batches=1)

    vols = [v for _, v in s.trace]
    assert all(a >= b - 1e-9 for a, b in zip(vols, vols[1:]))
    trials = [t for t, _ in s.trace]
    assert trials == sorted(trials)


def test_rejects_invalid_n():
    rng = np.random.default_rng(4)
    pts = _point_cloud(rng)
    with pytest.raises(ValueError):
        Search(pts, n=3, device="cpu")
    with pytest.raises(ValueError):
        Search(pts, n=1000, device="cpu")


def test_graph_capture_matches_ungraphed():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(6)
    pts = _point_cloud(rng)

    s1 = Search(pts, n=10, seed=42, batch_size=1024, device="cuda:0")
    res1 = s1.run(1024 * 10)

    s2 = Search(pts, n=10, seed=42, batch_size=1024, device="cuda:0")
    res2 = s2.run(1024 * 10, use_graph=True)

    assert res2.index == res1.index
    assert res2.volume == pytest.approx(res1.volume, rel=1e-4)
    np.testing.assert_allclose(res2.directions, res1.directions, atol=1e-5)


def test_graph_capture_mixed_with_ungraphed_batches_stays_consistent():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(7)
    pts = _point_cloud(rng)

    s_ref = Search(pts, n=8, seed=1, batch_size=512, device="cuda:0")
    res_ref = s_ref.run(512 * 12)

    s = Search(pts, n=8, seed=1, batch_size=512, device="cuda:0")
    s.run(512 * 4)
    res = s.run(512 * 8, use_graph=True)

    assert res.index == res_ref.index
    assert res.volume == pytest.approx(res_ref.volume, rel=1e-4)


def test_gpu_matches_cpu_search():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(5)
    pts = _point_cloud(rng)

    s_cpu = Search(pts, n=8, seed=5, batch_size=1024, device="cpu")
    res_cpu = s_cpu.run(4096)

    s_gpu = Search(pts, n=8, seed=5, batch_size=1024, device="cuda:0")
    res_gpu = s_gpu.run(4096)

    assert res_gpu.index == res_cpu.index
    assert res_gpu.volume == pytest.approx(res_cpu.volume, rel=1e-4)
    np.testing.assert_allclose(res_gpu.directions, res_cpu.directions, rtol=1e-4, atol=1e-5)
