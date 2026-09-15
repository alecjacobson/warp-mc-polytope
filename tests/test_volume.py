import numpy as np
import pytest

from tests.conftest import volume_of, batch_volumes

CUBE_DIRS = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
)


def test_cube():
    b = np.ones(6)
    v = volume_of(CUBE_DIRS, b, big=8.0)
    assert v == pytest.approx(8.0, rel=1e-5)


def test_octahedron():
    dirs = []
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                d = np.array([sx, sy, sz], dtype=np.float64)
                dirs.append(d / np.linalg.norm(d))
    dirs = np.array(dirs)
    b = np.ones(8) / np.sqrt(3)
    v = volume_of(dirs, b, big=8.0)
    assert v == pytest.approx(4.0 / 3.0, rel=1e-5)


def test_scaled_cube_scales_as_cube_of_side():
    for s in (0.5, 1.0, 2.0, 7.3):
        b = s * np.ones(6)
        v = volume_of(CUBE_DIRS, b, big=8.0 * s)
        assert v == pytest.approx((2.0 * s) ** 3, rel=1e-5)


def test_tetrahedron_closed_form():
    # Regular tetrahedron with vertices at alternating cube corners; its
    # face-normal halfspace representation has a known closed-form volume.
    dirs = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
    )
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    b = np.full(4, 1.0 / np.sqrt(3))
    v = volume_of(dirs, b, big=8.0)
    # Edge length of this tetrahedron is 2*sqrt(2); V = edge^3 / (6*sqrt(2)).
    edge = 2 * np.sqrt(2)
    expected = edge**3 / (6 * np.sqrt(2))
    assert v == pytest.approx(expected, rel=1e-4)


def test_unbounded_returns_inf():
    # Only 4 halfspaces bounding x and y but not z: unbounded along z.
    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    b = np.ones(4)
    v = volume_of(dirs, b, big=50.0)
    assert not np.isfinite(v)


def test_random_polytopes_match_pchs_reference():
    igl = pytest.importorskip("igl")
    pchs = pytest.importorskip("pchs")

    def reference_volume(dirs, b):
        halfspaces = np.hstack([dirs, -b.reshape(-1, 1)])
        x0 = pchs.chebyshev_center(halfspaces)
        rV, rPI, rPC = pchs.primal_mesh_from_halfspaces(halfspaces, x0)
        # primal_mesh_from_halfspaces does a naive polar-dual construction
        # that does not itself validate boundedness: for a genuinely
        # unbounded input it can silently return a bogus finite polytope.
        # Reject any reference whose vertices violate a constraint.
        tol = 1e-6 * (1.0 + np.max(np.abs(b)))
        if np.any(dirs @ rV.T > b.reshape(-1, 1) + tol):
            raise ValueError("reference infeasible (input likely unbounded)")
        rF, _ = igl.polygons_to_triangles(rPI, rPC)
        m0, _, _ = igl.moments(rV, rF)
        return m0

    rng = np.random.default_rng(0)
    n_tested = 0
    max_rel_err = 0.0
    for n in (4, 6, 8, 12, 20, 32):
        dirs_list, b_list, bigs, refs = [], [], [], []
        for _ in range(15):
            dirs = rng.standard_normal((n, 3))
            dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
            pts = rng.standard_normal((200, 3))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            pts *= 0.5 + 0.5 * rng.random((200, 1))
            b = np.max(dirs @ pts.T, axis=1)
            try:
                ref = reference_volume(dirs, b)
            except Exception:
                continue
            if not np.isfinite(ref) or ref <= 0:
                continue
            dirs_list.append(dirs)
            b_list.append(b)
            bigs.append(60.0)
            refs.append(ref)

        if not dirs_list:
            continue
        vols = batch_volumes(dirs_list, b_list, bigs)
        for v, ref in zip(vols, refs):
            n_tested += 1
            assert np.isfinite(v), f"unexpectedly flagged invalid (n={n}, ref={ref})"
            rel_err = abs(v - ref) / ref
            max_rel_err = max(max_rel_err, rel_err)
            assert rel_err < 1e-4, f"n={n} warp={v} ref={ref} rel_err={rel_err}"

    assert n_tested > 50
    print(f"tested {n_tested} random polytopes, max relative error = {max_rel_err:.3e}")


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_gpu_matches_cpu(gpu_device, n):
    rng = np.random.default_rng(n)
    v_cpu = float("inf")
    # n=4 directions bound a finite region only ~1/8 of the time (Wendel's
    # theorem: P(0 in conv of 4 symmetric random points in R^3) = 1/8), so
    # give this enough tries to reliably find one.
    for _ in range(300):
        dirs = rng.standard_normal((n, 3))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        pts = rng.standard_normal((200, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        b = np.max(dirs @ pts.T, axis=1)
        v_cpu = volume_of(dirs, b, big=60.0, device="cpu")
        if np.isfinite(v_cpu):
            break
    assert np.isfinite(v_cpu), "failed to draw a bounded random polytope in 300 tries"
    v_gpu = volume_of(dirs, b, big=60.0, device=gpu_device)
    assert v_gpu == pytest.approx(v_cpu, rel=1e-4)
