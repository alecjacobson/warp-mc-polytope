import argparse
import csv
import sys
import time

import numpy as np
import warp as wp


def _load_hull(mesh_path):
    """Read a mesh and return its full (non-simplified) convex hull (hV, hF)."""
    import igl
    import igl.copyleft.cgal

    V, F = igl.read_triangle_mesh(mesh_path)
    hF = igl.copyleft.cgal.convex_hull(V)
    hV, hF, _, _ = igl.remove_unreferenced(V, hF)
    return hV, hF


def _load_points(hV, hF, simplify_to):
    """(Optionally) pre-simplify a hull with PCHS to `simplify_to`
    vertices. Returns the point set every search trial's halfspaces must
    contain."""
    if simplify_to is None or simplify_to >= hV.shape[0]:
        return hV

    import pchs

    pV, pPI, pPC = pchs.simplify_convex_hull(hV, hF, simplify_to)
    return pV


def _load_candidates(hV, hF, weight_by):
    from mcpolytope.weights import WEIGHT_FUNCS, hull_face_candidates

    normals, areas = hull_face_candidates(hV, hF)
    weight_fn = WEIGHT_FUNCS[weight_by]
    weights = weight_fn(hV, hF, normals, areas)
    return normals, weights


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mcpolytope",
        description="GPU Monte Carlo search for the minimum-volume n-halfspace polytope enclosing a mesh's convex hull.",
    )
    parser.add_argument("mesh", help="input mesh file (.ply, .obj, ...)")
    parser.add_argument("-n", type=int, default=20, help="number of halfspaces/faces (default: 20)")
    parser.add_argument("--trials", type=float, default=1e7, help="number of Monte Carlo trials (default: 1e7)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1 << 20)
    parser.add_argument("--big-factor", type=float, default=64.0, help="bounding-square half-width / point radius")
    parser.add_argument(
        "--simplify-to",
        type=int,
        default=500,
        help="pre-simplify the mesh's convex hull to this many vertices via PCHS before searching (default: 500; 0 disables)",
    )
    parser.add_argument("--device", default=None, help="warp device (default: preferred, e.g. cuda:0)")
    parser.add_argument("--use-graph", action="store_true", help="use CUDA graph capture for the trial loop")
    parser.add_argument("--trace-csv", default=None, help="write (trials, best_volume) checkpoints to this CSV")
    parser.add_argument(
        "--trace-every-batches", type=int, default=None, help="record a trace checkpoint every this many batches"
    )
    parser.add_argument(
        "--candidates",
        choices=["random", "original-hull"],
        default="random",
        help="sample directions uniformly over the sphere (default), or from the mesh's original "
        "(pre-simplification) convex hull face normals",
    )
    parser.add_argument(
        "--weight-by",
        choices=["uniform", "area", "inverse_area", "solid_angle", "dihedral"],
        default="uniform",
        help="candidate weighting scheme (only meaningful with --candidates original-hull)",
    )
    parser.add_argument(
        "--avoid-duplicates", action="store_true", help="avoid resampling the same candidate direction within a trial"
    )
    args = parser.parse_args(argv)

    wp.init()

    from mcpolytope.search import Search

    hV, hF = _load_hull(args.mesh)
    simplify_to = args.simplify_to if args.simplify_to > 0 else None
    points = _load_points(hV, hF, simplify_to)
    print(f"loaded {points.shape[0]} points from {args.mesh}", file=sys.stderr)

    candidates = weights = None
    if args.candidates == "original-hull":
        candidates, weights = _load_candidates(hV, hF, args.weight_by)
        print(f"using {candidates.shape[0]} original-hull candidate directions, weight_by={args.weight_by}", file=sys.stderr)

    search = Search(
        points,
        n=args.n,
        seed=args.seed,
        big_factor=args.big_factor,
        batch_size=args.batch_size,
        device=args.device,
        candidates=candidates,
        weights=weights,
        avoid_duplicates=args.avoid_duplicates,
    )

    trace_every = args.trace_every_batches
    if args.trace_csv and trace_every is None:
        trace_every = 1

    t0 = time.perf_counter()
    result = search.run(int(args.trials), trace_every_batches=trace_every, use_graph=args.use_graph)
    if str(search.device).startswith("cuda"):
        wp.synchronize_device(search.device)
    dt = time.perf_counter() - t0

    print(f"trials: {search.trials_run:,}  time: {dt:.3f}s  ({search.trials_run / dt:,.0f} trials/sec)")
    print(f"best volume: {result.volume!r}  (trial index {result.index})")

    if args.trace_csv:
        with open(args.trace_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trials", "best_volume"])
            w.writerows(result.trace)
        print(f"wrote trace to {args.trace_csv}", file=sys.stderr)

    return result


if __name__ == "__main__":
    main()
