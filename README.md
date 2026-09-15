# warp-mc-polytope

GPU Monte Carlo search, in pure [NVIDIA Warp](https://github.com/NVIDIA/warp), for the
minimum-volume convex polytope with `n` faces that encloses a point set — built by
sampling huge numbers of random `n`-direction sets and, for each, computing the volume
of the resulting halfspace intersection **entirely on-device**.

Status: core search is implemented, tested (accuracy/robustness/GPU-CPU parity), and
benchmarked on an NVIDIA L40. See [Results](#results) below.

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

### Search driver: no host sync, deterministic replay, optional CUDA graph

`mcpolytope/search.py`'s `Search` class runs trials in batches (default `1<<20`
trials/batch). Each batch is seeded deterministically from `(seed, batch_index)`, so
rather than storing every trial's directions (infeasible at billions of trials), the
winning trial's directions are *regenerated* after the fact from its `(batch_index,
local_index)`. Cross-batch reduction (running best volume + its global index) happens
entirely device-side via two passes per batch:

1. `atomic_min` every trial's volume into a persistent `global_best_vol` device scalar.
2. Any trial whose volume now equals `global_best_vol` claims the global-best index via
   `atomic_exch` (**not** `atomic_min` — a later batch's win can have a *larger* global
   index than an earlier, now-stale one, which a monotonic `atomic_min` could never
   adopt; this was an actual bug caught by testing, see git history).

`run(..., use_graph=True)` captures one full batch step (trial kernel + both reduction
passes + a device-side batch-index increment) into a CUDA graph and replays it,
verified to produce bit-identical results to the ungraphed path. On the L40 at
`batch_size >= 1<<20` we're already compute-bound (<1% difference either way) — the
option is there for smaller-batch / launch-overhead-bound scenarios.

## Layout

```
mcpolytope/
  support.py   # wp.func: support offsets b_i = max_j dot(dir_i, points_j)
  volume.py    # wp.func: polytope_volume_from_halfspaces (facet-clipping algorithm)
  search.py    # Search: batched trial + on-device reduction + optional CUDA graph
  cli.py       # `mcpolytope mesh.ply -n 20 --trials 1e9 ...`
tests/         # accuracy (vs pchs reference), robustness, GPU/CPU parity, perf smoke
examples/
  actaeon_search.py  # compares this search against PCHS's greedy simplification
```

## Usage

```
pip install -e ".[test]"
mcpolytope path/to/mesh.ply -n 20 --trials 1e9 --use-graph --trace-csv trace.csv
```

or, to compare directly against PCHS's greedy simplification for the same `n`:

```
python examples/actaeon_search.py path/to/mesh.ply -n 8 --trials 5e8
```

## Running tests

```
pytest
```

## Results

Measured on an NVIDIA L40, mesh `Actaeon.ply` (66k faces) reduced to a 996-point convex
hull, then PCHS-pre-simplified to 500 points before searching (matching the point set
PCHS's own greedy simplification competes over):

| n  | trials      | time   | trials/sec | MC search volume | PCHS greedy volume | MC / PCHS |
|----|-------------|--------|------------|-------------------|---------------------|-----------|
| 8  | 5.0e8       | 11.0s  | 45,600,000 | 1.13706           | 0.84954             | 1.34x     |
| 20 | 2.0e9       | 224.4s | 8,912,976  | 0.88935           | 0.71450 *           | 1.24x *   |

(\* PCHS greedy volume for n=20 taken from a separate simplification run, not
re-measured in this table's session.)

**Takeaway:** unstructured random-direction search, even at billions of trials, does
not beat PCHS's greedy dual-edge-collapse simplification for either face count tested.
The search's own trace (`--trace-csv`) shows fast early improvement that decays
logarithmically — consistent with a `3n`-dimensional continuous optimization landscape
where good direction sets occupy a small volume fraction that unstructured Monte Carlo
struggles to concentrate on, while greedy edge collapse exploits local mesh structure
directly. This is a real (if somewhat expected) negative result for "does brute force
random search compete with a good heuristic here" — smaller `n` and/or smarter sampling
(e.g. importance sampling around greedy solutions, simulated annealing / local search
instead of i.i.d. resampling) are the natural next things to try if closing this gap
mattered.
