import numpy as np
import pytest
import warp as wp

from mcpolytope.support import support_offset

wp.init()


@wp.kernel
def _support_kernel(
    dirs: wp.array(dtype=wp.float32, ndim=2),
    points: wp.array(dtype=wp.vec3),
    m: int,
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = support_offset(dirs[i, 0], dirs[i, 1], dirs[i, 2], points, m)


def support_offsets(dirs, points, device="cpu"):
    n = len(dirs)
    m = len(points)
    out = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        _support_kernel,
        dim=n,
        inputs=[
            wp.array(dirs.astype(np.float32), dtype=wp.float32, device=device),
            wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
            m,
            out,
        ],
        device=device,
    )
    return out.numpy()


def test_matches_numpy_brute_force():
    rng = np.random.default_rng(0)
    points = rng.standard_normal((500, 3))
    dirs = rng.standard_normal((64, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    got = support_offsets(dirs, points)
    expected = np.max(dirs @ points.T, axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_centroid_shifted_points_have_positive_support():
    rng = np.random.default_rng(1)
    points = rng.standard_normal((300, 3)) * (0.5 + rng.random((300, 1)))
    points -= points.mean(axis=0)
    dirs = rng.standard_normal((128, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    b = support_offsets(dirs, points)
    assert np.all(b > 0)


def test_gpu_matches_cpu(gpu_device=None):
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    rng = np.random.default_rng(2)
    points = rng.standard_normal((400, 3))
    dirs = rng.standard_normal((32, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cpu = support_offsets(dirs, points, device="cpu")
    gpu = support_offsets(dirs, points, device="cuda:0")
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)
