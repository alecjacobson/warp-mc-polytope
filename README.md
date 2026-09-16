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

### Sampling from a weighted candidate direction set

Sampling directions i.i.d. from the continuous sphere turns out to converge far too
slowly to compete with PCHS's greedy simplification (see the first row of
[Results](#results)). `Search(..., candidates=(K,3), weights=(K,)|None,
avoid_duplicates=False)` instead draws each trial's `n` directions from a **finite
pool** — e.g. the face normals of the mesh's *original* (non-simplified) convex hull —
optionally weighted (`mcpolytope/weights.py`: `area`, `inverse_area`, a cheap
`solid_angle` proxy, `dihedral`/sharp-edge score). This concentrates the search on
directions that are geometrically plausible real facets instead of wasting trials on
directions nothing in the mesh actually points toward.

- **Sampling is O(1) per draw** via [Walker's alias method](https://en.wikipedia.org/wiki/Alias_method)
  (`mcpolytope/candidates.py`), independent of pool size `K` — matters since this runs
  `n` times per trial across billions of trials. (Caught a real bug here: a `wp.uint32`
  RNG state passed into a `wp.func` is *not* mutated-in-place across the call boundary —
  only sequential builtin calls within one function body are — so failing to
  reassign `state` from `sample_alias`'s return made every draw in a trial identical.
  See git history / `test_successive_draws_within_one_thread_are_not_all_identical`.)
- **With replacement by default** (cheap); `avoid_duplicates=True` opts into a
  bounded-retry rejection scheme (falls back to a deterministic linear scan for the
  first unused candidate on repeated collision — never an infinite loop, requires
  `n <= K`) rather than a true weighted-reservoir sampler, since collisions are rare
  when `K >> n` and a full weighted-without-replacement sampler would cost `O(K)` per
  trial for no measured benefit at that regime.
- This also exposed a second, more fundamental bug in `polytope_volume_from_halfspaces`
  itself: the degenerate-constraint branch had no numerical tolerance, so float32
  rounding noise in `dot(d,d)` for (near-)duplicate directions could push a
  mathematically-redundant constraint's `rhs` a hair negative, wrongly zeroing out an
  entire facet (observed as an 8-identical-directions trial silently reporting volume
  `0.0` instead of the correct `+inf`). This was a latent gap continuous Gaussian
  sampling could never trigger (near-zero probability of exact/near-duplicate
  directions there) — only surfaced once a finite candidate pool made duplicates common.

### Continuous importance sampling instead of subset selection

Discrete candidate sampling can *only* ever pick one of the mesh's existing `K` face
normals — never a direction "between" two real faces (e.g. a bevel) that might support a
tighter polytope. `Search(..., candidates=, weights=, kappa=)` instead uses the
weighted candidate distribution purely as an **importance-sampling signal**: each draw
picks a seed candidate via the alias method, then samples the *actual* trial direction
from a continuous [von Mises-Fisher](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)
kernel (`mcpolytope/candidates.py::sample_vmf`) centered on that seed with concentration
`kappa` — `kappa=0` ignores the weighting entirely (uniform sphere), `kappa -> inf`
recovers exact discrete sampling, intermediate `kappa` gives a tunable "fuzziness" that
lets a trial land on and score directions no input face has. The S² case has an exact,
closed-form, rejection-free sampler (unlike the general Wood-1994 algorithm needed for
higher dimensions — see the module docstring), so this is as cheap as the discrete path.

## Layout

```
mcpolytope/
  support.py     # wp.func: support offsets b_i = max_j dot(dir_i, points_j)
  volume.py      # wp.func: polytope_volume_from_halfspaces (facet-clipping algorithm)
  candidates.py  # O(1) weighted categorical sampling (Walker's alias method)
  weights.py     # mesh -> candidate directions + weight schemes (area, dihedral, ...)
  search.py      # Search: batched trial + on-device reduction + optional CUDA graph
  cli.py         # `mcpolytope mesh.ply -n 20 --trials 1e9 ...`
tests/           # accuracy (vs pchs reference), robustness, GPU/CPU parity, perf smoke
examples/
  actaeon_search.py    # compares random-sphere search against PCHS's greedy simplification
  weight_experiment.py # compares all candidate weighting schemes (+ avoid_duplicates) against PCHS
```

## Usage

```
pip install -e ".[test]"
mcpolytope path/to/mesh.ply -n 20 --trials 1e9 --use-graph --trace-csv trace.csv
```

restricting to the mesh's original hull face normals, area-weighted, without
duplicates within a trial (see [Results](#results) for why this combination matters --
it's the best-performing configuration found so far):

```
mcpolytope path/to/mesh.ply -n 20 --trials 1e8 --candidates original-hull --weight-by area --avoid-duplicates
```

or as a continuous importance-sampling signal instead of exact discrete reuse (see
[Results](#results) for why, on the meshes tested, this is *not* currently an
improvement over the discrete version above):

```
mcpolytope path/to/mesh.ply -n 20 --trials 1e8 --candidates original-hull --weight-by area --kappa 1000
```

or, to compare directly against PCHS's greedy simplification for the same `n`:

```
python examples/actaeon_search.py path/to/mesh.ply -n 8 --trials 5e8
python examples/weight_experiment.py path/to/mesh.ply -n 20 --trials 1e8
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

**Takeaway (random-sphere sampling):** unstructured random-direction search over the
*continuous* sphere, even at billions of trials, does not beat PCHS's greedy
dual-edge-collapse simplification for either face count tested. The search's own trace
(`--trace-csv`) shows fast early improvement that decays logarithmically — consistent
with a `3n`-dimensional continuous optimization landscape where good direction sets
occupy a small volume fraction that unstructured Monte Carlo struggles to concentrate
on, while greedy edge collapse exploits local mesh structure directly.

### Candidate-direction weighting comparison

That negative result motivated restricting the search to a **finite candidate pool**
(the mesh's original, pre-simplification hull face normals) instead of the continuous
sphere (`examples/weight_experiment.py`, same `Actaeon.ply` setup as above, `n=20`,
1e8 trials/scheme on the L40; PCHS greedy for `n=20` is `0.71448`):

| scheme (weight_by) | avoid_duplicates | volume  | / PCHS |
|---------------------|:---:|---------|--------|
| (random sphere, for reference) | n/a | 0.90454 | 1.266x |
| uniform              | either | 0.80509 | 1.127x |
| dihedral             | either | 0.80922 | 1.133x |
| inverse_area         | either | 0.93260 | 1.305x |
| solid_angle          | False | 0.74550 | 1.043x |
| **solid_angle**      | **True** | **0.71777** | **1.005x** |
| area                 | False | 0.72035 | 1.008x |
| **area**             | **True** | **0.71564** | **1.002x** |

And at `n=8` (1e7 trials/scheme; PCHS greedy is `0.84954`), **area-weighted candidate
search with `avoid_duplicates=True` actually beats PCHS greedy**, `0.83249` vs
`0.84954` — 2% smaller:

| scheme (weight_by) | avoid_duplicates | volume  | / PCHS |
|---------------------|:---:|---------|--------|
| (random sphere, for reference) | n/a | 1.20511 | 1.419x |
| **area**            | **either** | **0.83249** | **0.980x** |
| solid_angle          | True | 0.83611 | 0.984x |
| uniform / dihedral   | either | ~1.04 | ~1.22x |
| inverse_area         | either | ~1.26 | ~1.49x |

**Takeaway (candidate sampling):** restricting to real hull-face directions, weighted
by **area**, closes almost the entire gap to (and at `n=8`, beats) PCHS's greedy
simplification — a striking difference from unstructured continuous sampling, at a
tiny fraction of the trial budget (1e7-1e8 vs 5e8-2e9). `inverse_area` performs worst,
consistent with small/sliver faces rarely being useful supporting directions for a
tight enclosure. `avoid_duplicates=True` only matters (and only barely) for `area` and
`solid_angle` — the two most concentrated/skewed weightings, where without-replacement
sampling meaningfully reduces wasted duplicate-direction trials; for flatter
distributions (`uniform`, `dihedral`, `inverse_area`) collisions are already rare
enough at `K` in the thousands that it made no measurable difference, matching the "only
implement if it measurably helps" premise this feature started from — so it's an
available option, not the default. Smaller `n` and/or smarter sampling (e.g. importance
sampling seeded from a greedy solution, simulated annealing / local search instead of
i.i.d. resampling) remain the natural next things to try if closing the remaining gap
at larger `n` mattered.

### Continuous importance sampling (kappa) vs. subset selection

The obvious question: can allowing directions the mesh doesn't actually have — via
continuous vMF jitter around the weighted candidates instead of exact reuse — beat
discrete subset selection? Swept `kappa` for the `area` and `solid_angle` schemes at
both `n=20` (1e8 trials/kappa) and `n=8` (5e7 trials/kappa):

`n=20`, `area` scheme (discrete exact, `avoid_duplicates=True`: `0.71499`, `1.0007x`):

| kappa | volume  | / PCHS |
|------:|---------|--------|
| 50    | 0.82421 | 1.154x |
| 200   | 0.78873 | 1.104x |
| 1000  | 0.75994 | 1.064x |
| 5000  | 0.74325 | 1.040x |

`n=8`, `area` scheme (discrete exact, `avoid_duplicates=True`: `0.83249`, `0.980x`):

| kappa | volume  | / PCHS |
|------:|---------|--------|
| 0     | 1.02801 | 1.210x |
| 50    | 0.98561 | 1.160x |
| 200   | 0.91079 | 1.072x |
| 1000  | 0.88656 | 1.044x |
| 5000  | 0.86223 | 1.015x |

**Takeaway (continuous importance sampling): no — at every kappa tried, at matched
trial budgets, continuous vMF jitter around the weighted candidates converges *toward*
the discrete result but never reaches or beats it**, at either `n`. The trend is
monotonic and smooth in `kappa` (validating the sampler: `kappa->inf` really does
recover the discrete answer, as designed), it just never crosses over even at `kappa`
high enough that samples are typically within ~1 degree of their seed face. The likely
reason: `Actaeon.ply` is a real physical mesh with genuinely flat faces, so its actual
face normals already *are* the best local supporting directions for a tight
enclosure — a small random perturbation off a real (already near-optimal) face normal
is a strictly worse direction far more often than a better one, so continuous jitter
mostly just adds noise. Reaching a genuinely novel direction (e.g. a bevel exactly
between two adjacent real faces) would need either much higher trial budgets than
tested here, or a smarter proposal than a single-seed symmetric vMF kernel (e.g. a
kernel centered on edges/bevels between adjacent weighted faces rather than face
centers) — a reasonable next thing to try, but out of scope for this round: for this
mesh, at these budgets, **subset selection (discrete candidate sampling) remains the
better strategy**.
