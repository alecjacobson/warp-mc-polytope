# warp-mc-polytope

GPU Monte Carlo search, in pure [NVIDIA Warp](https://github.com/NVIDIA/warp), for the
minimum-volume convex polytope with `n` faces that encloses a point set — built by
sampling huge numbers of random `n`-direction sets and, for each, computing the volume
of the resulting halfspace intersection **entirely on-device**.

Status: work in progress. See below for what's implemented so far.

## The problem

Given a point set `P` (e.g. a mesh's convex hull, possibly pre-simplified) and a target
face count `n`, find `n` unit directions `d_1..d_n` minimizing the volume of

```
polytope(d) = { x : d_i . x <= b_i(d_i),  i = 1..n }     where  b_i(d_i) = max_{p in P} d_i . p
```

(`b_i` is the support value of `P` along `d_i`, so `polytope(d)` always contains `P` —
it's a valid, if not necessarily optimal, `n`-face outer approximation of `conv(P)`.)
This is a competitor/baseline for greedy dual-edge-collapse simplification
([PCHS](https://github.com/alecjacobson/progressive-convex-hull-simplification)):
does massive random search ever find a smaller-volume `n`-hedron, especially for small
`n`?

## Why this is nontrivial on a GPU

Generating random directions and support values (`b_i`) is embarrassingly parallel and
cheap. The hard part is computing `volume(polytope(d))` for an arbitrary set of `n`
halfspaces **inside a Warp kernel**, with no dynamic allocation, no per-trial LP solve,
and no host sync — needed because the search only pays off at millions-to-billions of
trials per second.

### No per-trial LP for an interior point

Shift all points by their centroid once, on the host, before launching any kernels.
The centroid is strictly interior to `conv(P)`, so for *any* unit direction `d`,
`max_p d.(p - centroid) > 0`. Building halfspaces from support values of the
recentered points therefore guarantees the origin is strictly interior to every
sampled polytope — no [Chebyshev-center](https://en.wikipedia.org/wiki/Chebyshev_center)
LP needed per trial, just the `O(n*m)` support computation.

### Volume of an H-representation via per-facet half-plane clipping

`mcpolytope/volume.py`'s `polytope_volume_from_halfspaces` computes the volume of an
`n`-halfspace intersection (with the origin known interior) without ever building a
dual convex hull. For each facet `i`:

1. Build a local 2D orthonormal basis `(U, V)` perpendicular to `d_i`.
2. Every other constraint `j` restricted to facet `i`'s plane becomes a 2D half-plane
   `(d_j.U) u + (d_j.V) v <= b_j - b_i (d_j.d_i)`.
3. Starting from a large square (side controlled by a `big` parameter — set from the
   point cloud's bounding radius), clip against all `n-1` other constraints
   ([Sutherland–Hodgman](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm)).
4. Shoelace-area the resulting 2D polygon and accumulate
   `vol += (1/3) * b_i * area_i` (divergence theorem — the distance from the origin to
   plane `i` is exactly `b_i` since `d_i` is unit).

This is `O(n)` per facet, `O(n^2)` per facet-clip, `O(n^3)` total — the same complexity
class as Lasserre-style exact-volume methods, but implementable with only fixed-size
per-thread local arrays (`n <= N_MAX = 32`), which is what makes it Warp/register-friendly.

**Unboundedness / degenerate direction sets:** small `n` direction sets don't always
positively span R^3 (e.g. Wendel's theorem: 4 random symmetric directions bound a
finite region only 1/8 of the time), in which case the "polytope" is actually
unbounded. This is detected — not silently mis-answered — by checking whether any
surviving facet-polygon vertex reaches the boundary of the initial bounding square;
such trials return `+inf` so they're never selected as the search minimum.

Validated against [`pchs`](https://github.com/alecjacobson/progressive-convex-hull-simplification)'s
CGAL/qhull-backed dual-hull construction across `n in {4,6,8,12,20,32}`: max relative
error ~1e-6 on bounded random polytopes (see `tests/test_volume.py`).

## Layout

```
mcpolytope/
  support.py   # wp.func: support offsets b_i = max_j dot(dir_i, points_j)
  volume.py    # wp.func: polytope_volume_from_halfspaces (facet-clipping algorithm)
  search.py    # search driver (in progress)
tests/         # accuracy (vs pchs reference), robustness, GPU/CPU parity
```

## Running tests

```
pip install -e ".[test]"
pytest
```
