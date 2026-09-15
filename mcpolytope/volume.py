import warp as wp

# Maximum number of halfspaces/directions a single trial can use. Kept as a
# fixed compile-time bound so all per-thread storage is register-resident
# (no dynamic/heap allocation inside kernels).
N_MAX = 32

# A facet-local 2D clip polygon can gain at most one vertex per half-plane
# clip (Sutherland-Hodgman clipping a convex polygon by one half-plane).
# Starting from a 4-gon and clipping against up to N_MAX-1 other
# constraints, the vertex count is bounded by N_MAX + 3.
POLY_MAX = N_MAX + 4

fvecN = wp.types.vector(length=N_MAX, dtype=wp.float32)
polyvec = wp.types.vector(length=POLY_MAX, dtype=wp.float32)

# Relative tolerances (multiplied by the bounding-square half-width `big`).
_DEGENERATE_REL_TOL = 1.0e-7
_BOUNDARY_REL_TOL = 1.0e-4


@wp.func
def _clip_halfplane(px: polyvec, py: polyvec, count: int, a_u: float, a_v: float, rhs: float):
    """Sutherland-Hodgman clip of convex polygon (px,py)[:count] by a_u*u + a_v*v <= rhs."""
    new_px = polyvec()
    new_py = polyvec()
    new_count = int(0)
    for k in range(count):
        k2 = k + 1
        if k2 == count:
            k2 = 0
        x1 = px[k]
        y1 = py[k]
        x2 = px[k2]
        y2 = py[k2]
        d1 = a_u * x1 + a_v * y1 - rhs
        d2 = a_u * x2 + a_v * y2 - rhs
        in1 = d1 <= 0.0
        in2 = d2 <= 0.0
        if in1:
            if new_count < POLY_MAX:
                new_px[new_count] = x1
                new_py[new_count] = y1
                new_count += 1
        if in1 != in2:
            t = d1 / (d1 - d2)
            xi = x1 + t * (x2 - x1)
            yi = y1 + t * (y2 - y1)
            if new_count < POLY_MAX:
                new_px[new_count] = xi
                new_py[new_count] = yi
                new_count += 1
    return new_px, new_py, new_count


@wp.func
def _polygon_area(px: polyvec, py: polyvec, count: int):
    area = float(0.0)
    for k in range(count):
        k2 = k + 1
        if k2 == count:
            k2 = 0
        area += px[k] * py[k2] - px[k2] * py[k]
    return 0.5 * wp.abs(area)


@wp.func
def polytope_volume_from_halfspaces(dx: fvecN, dy: fvecN, dz: fvecN, b: fvecN, n: int, big: float):
    """Volume of {x : dir_i . x <= b_i, i in [0,n)}, assuming (dx,dy,dz)_i are
    unit normals and the origin is strictly interior (all b_i > 0).

    Computed by clipping, per facet, a large square in the facet's local 2D
    frame against the traces of all other constraints (Sutherland-Hodgman),
    then summing (1/3) * b_i * area_i (divergence theorem). Returns +inf if
    the polytope is unbounded (or `big` is too small to safely bound it) --
    detected when a surviving facet polygon vertex reaches the initial
    bounding square's boundary.
    """
    vol = float(0.0)
    invalid = bool(False)
    eps = _BOUNDARY_REL_TOL * big
    deg_tol = _DEGENERATE_REL_TOL * big

    for i in range(n):
        if not invalid:
            nix = dx[i]
            niy = dy[i]
            niz = dz[i]

            if wp.abs(nix) < 0.9:
                tx = float(1.0)
                ty = float(0.0)
                tz = float(0.0)
            else:
                tx = float(0.0)
                ty = float(1.0)
                tz = float(0.0)

            ux = niy * tz - niz * ty
            uy = niz * tx - nix * tz
            uz = nix * ty - niy * tx
            ulen = wp.sqrt(ux * ux + uy * uy + uz * uz)
            ux = ux / ulen
            uy = uy / ulen
            uz = uz / ulen

            vx = niy * uz - niz * uy
            vy = niz * ux - nix * uz
            vz = nix * uy - niy * ux

            px = polyvec()
            py = polyvec()
            px[0] = -big
            py[0] = -big
            px[1] = big
            py[1] = -big
            px[2] = big
            py[2] = big
            px[3] = -big
            py[3] = big
            count = int(4)

            bi = b[i]
            for j in range(n):
                if j != i and count > 0:
                    njx = dx[j]
                    njy = dy[j]
                    njz = dz[j]
                    a_u = njx * ux + njy * uy + njz * uz
                    a_v = njx * vx + njy * vy + njz * vz
                    dotij = nix * njx + niy * njy + niz * njz
                    rhs = b[j] - bi * dotij
                    norm_a = wp.sqrt(a_u * a_u + a_v * a_v)
                    if norm_a < deg_tol:
                        if rhs < 0.0:
                            count = 0
                    else:
                        px, py, count = _clip_halfplane(px, py, count, a_u, a_v, rhs)

            if count >= 3:
                touches = bool(False)
                for k in range(count):
                    if wp.abs(px[k]) > big - eps or wp.abs(py[k]) > big - eps:
                        touches = True
                if touches:
                    invalid = True
                else:
                    area = _polygon_area(px, py, count)
                    vol += (1.0 / 3.0) * bi * area

    if invalid:
        return wp.inf
    return vol
