import numpy as np


def hull_face_candidates(V, F):
    """Per-face outward unit normals and areas of a triangle mesh -- used as
    the candidate direction set (typically the mesh's *original*, non-
    simplified convex hull) and the basis for area-derived weights.
    """
    import igl

    normals = igl.per_face_normals(V, F, np.array([0.0, 0.0, 1.0]))
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / lengths
    areas = 0.5 * igl.doublearea(V, F)
    return normals, areas


def uniform_weight(V, F, normals, areas):
    return np.ones(len(areas))


def area_weight(V, F, normals, areas):
    return areas.copy()


def inverse_area_weight(V, F, normals, areas):
    eps = 1e-12 * max(areas.max(), 1e-12)
    return 1.0 / (areas + eps)


def solid_angle_weight(V, F, normals, areas):
    """Cheap planar-patch solid-angle proxy: area / distance(centroid, face)^2.

    Not an exact spherical-Voronoi solid angle -- just area attenuated by
    squared distance from the hull centroid, so faces that are both large
    and close to the centroid (i.e. subtend more of the direction sphere)
    get more weight.
    """
    centroid = V[np.unique(F)].mean(axis=0)
    face_centers = V[F].mean(axis=1)
    dist2 = np.sum((face_centers - centroid) ** 2, axis=1)
    eps = 1e-12 * max(dist2.max(), 1e-12)
    return areas / (dist2 + eps)


def dihedral_weight(V, F, normals, areas):
    """Per-face feature/sharp-edge score: the largest angle (radians)
    between this face's normal and any edge-adjacent face's normal. ~0 on
    flat/coplanar regions, larger near sharp edges.
    """
    import igl

    TT, _ = igl.triangle_triangle_adjacency(F)
    K = F.shape[0]
    weight = np.zeros(K)
    for f in range(K):
        best = 0.0
        for e in range(3):
            nbr = TT[f, e]
            if nbr < 0:
                continue
            cos_angle = np.clip(np.dot(normals[f], normals[nbr]), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            best = max(best, angle)
        weight[f] = best
    # isolated faces (no neighbors, degenerate input) get the mean of the
    # rest rather than 0, so they're not silently excluded from sampling.
    zero_mask = weight <= 0
    if zero_mask.any() and not zero_mask.all():
        weight[zero_mask] = weight[~zero_mask].mean()
    elif zero_mask.all():
        weight[:] = 1.0
    return weight


WEIGHT_FUNCS = {
    "uniform": uniform_weight,
    "area": area_weight,
    "inverse_area": inverse_area_weight,
    "solid_angle": solid_angle_weight,
    "dihedral": dihedral_weight,
}
