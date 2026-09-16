import numpy as np
import pytest

igl = pytest.importorskip("igl")

from mcpolytope.weights import (
    WEIGHT_FUNCS,
    area_weight,
    dihedral_weight,
    hull_face_candidates,
    inverse_area_weight,
    solid_angle_weight,
    uniform_weight,
)


def _cube_mesh(side=2.0):
    h = side / 2.0
    V = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ],
        dtype=np.float64,
    )
    F = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom (-z)
            [4, 5, 6], [4, 6, 7],  # top (+z)
            [0, 1, 5], [0, 5, 4],  # -y
            [1, 2, 6], [1, 6, 5],  # +x
            [2, 3, 7], [2, 7, 6],  # +y
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        dtype=np.int64,
    )
    return V, F


def _hinge_mesh(theta):
    """Two triangles sharing the edge (0,0,0)-(0,1,0); the second triangle's
    third vertex is rotated by `theta` (radians) away from the first
    triangle's plane. theta near pi -> nearly flat continuation; theta near
    pi/2 -> a sharp fold.
    """
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([np.cos(theta), 0.0, np.sin(theta)])
    V = np.array([v0, v1, v2, v3])
    F = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int64)
    return V, F


def test_hull_face_candidates_cube_areas_sum_to_surface_area():
    V, F = _cube_mesh(side=2.0)
    normals, areas = hull_face_candidates(V, F)
    assert normals.shape == (12, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-10)
    assert areas.sum() == pytest.approx(6 * (2.0 * 2.0), rel=1e-10)


def test_area_and_inverse_area_weight():
    V, F = _cube_mesh()
    normals, areas = hull_face_candidates(V, F)
    np.testing.assert_allclose(area_weight(V, F, normals, areas), areas)
    np.testing.assert_allclose(inverse_area_weight(V, F, normals, areas), 1.0 / areas, rtol=1e-6)


def test_uniform_weight_is_constant():
    V, F = _cube_mesh()
    normals, areas = hull_face_candidates(V, F)
    w = uniform_weight(V, F, normals, areas)
    assert np.all(w == w[0])


def test_solid_angle_weight_matches_cube_symmetry():
    V, F = _cube_mesh()
    normals, areas = hull_face_candidates(V, F)
    w = solid_angle_weight(V, F, normals, areas)
    # a cube is symmetric under its face group -> every face should get
    # (numerically) the same solid-angle-proxy weight.
    np.testing.assert_allclose(w, w[0], rtol=1e-8)


def test_dihedral_weight_larger_for_sharper_fold():
    V_flat, F_flat = _hinge_mesh(theta=np.deg2rad(179))
    V_sharp, F_sharp = _hinge_mesh(theta=np.deg2rad(90))

    n_flat, _ = hull_face_candidates(V_flat, F_flat)
    n_sharp, _ = hull_face_candidates(V_sharp, F_sharp)

    w_flat = dihedral_weight(V_flat, F_flat, n_flat, None)
    w_sharp = dihedral_weight(V_sharp, F_sharp, n_sharp, None)

    assert w_flat.shape == (2,)
    assert np.all(w_sharp > w_flat)


def test_dihedral_weight_handles_isolated_face():
    # single triangle, no neighbors: TT is all -1, weight must not be all-zero
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    F = np.array([[0, 1, 2]], dtype=np.int64)
    normals, _ = hull_face_candidates(V, F)
    w = dihedral_weight(V, F, normals, None)
    assert w.shape == (1,)
    assert w[0] > 0


def test_weight_funcs_registry_all_nonnegative_and_finite():
    V, F = _cube_mesh()
    normals, areas = hull_face_candidates(V, F)
    for name, fn in WEIGHT_FUNCS.items():
        w = fn(V, F, normals, areas)
        assert np.all(np.isfinite(w)), name
        assert np.all(w >= 0), name
        assert w.shape == (12,), name
