import argparse
import csv
import sys
import time

import numpy as np
import warp as wp


def _load_points(mesh_path, simplify_to):
    """Read a mesh, take its convex hull, and (optionally) pre-simplify it
    with PCHS to `simplify_to` vertices. Returns the point set every search
    trial's halfspaces must contain."""
    import igl
    import igl.copyleft.cgal

    V, F = igl.read_triangle_mesh(mesh_path)
    hF = igl.copyleft.cgal.convex_hull(V)
    hV, hF, _, _ = igl.remove_unreferenced(V, hF)

    if simplify_to is None or simplify_to >= hV.shape[0]:
        return hV

    import pchs

    pV, pPI, pPC = pchs.simplify_convex_hull(hV, hF, simplify_to)
    return pV


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
    args = parser.parse_args(argv)

    wp.init()

    from mcpolytope.search import Search

    simplify_to = args.simplify_to if args.simplify_to > 0 else None
    points = _load_points(args.mesh, simplify_to)
    print(f"loaded {points.shape[0]} points from {args.mesh}", file=sys.stderr)

    search = Search(
        points,
        n=args.n,
        seed=args.seed,
        big_factor=args.big_factor,
        batch_size=args.batch_size,
        device=args.device,
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
