import warp as wp


@wp.func
def support_offset(
    dirx: float,
    diry: float,
    dirz: float,
    points: wp.array(dtype=wp.vec3),
    m: int,
):
    """Support value b = max_j dot(dir, points[j]) for a single unit direction.

    Assumes the origin is strictly interior to conv(points) (e.g. points have
    already been re-centered on their centroid), so b is guaranteed > 0.
    """
    b = float(-1.0e30)
    for j in range(m):
        p = points[j]
        d = dirx * p[0] + diry * p[1] + dirz * p[2]
        if d > b:
            b = d
    return b
