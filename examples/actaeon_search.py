"""Port of convex-hull-simplification/warp-monte-carlo.py's workflow onto
mcpolytope.Search: compare PCHS's greedy simplification against a GPU Monte
Carlo random-direction search for the same face count `n`.

Usage:
    python examples/actaeon_search.py path/to/Actaeon.ply --n 20 --trials 1e8
"""

import argparse
import time

import igl
import igl.copyleft.cgal
import numpy as np
import pchs
import warp as wp

from mcpolytope.search import Search


def triangle_mesh_volume(V, F):
    m0, _, _ = igl.moments(V, F)
    return m0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--trials", type=float, default=1e8)
    parser.add_argument("--simplify-to", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    wp.init()

    V, F = igl.read_triangle_mesh(args.mesh)
    hF = igl.copyleft.cgal.convex_hull(V)
    hV, hF, _, _ = igl.remove_unreferenced(V, hF)
    print(f"vol(hull) = {triangle_mesh_volume(hV, hF):g}")

    pV, pPI, pPC = pchs.simplify_convex_hull(hV, hF, args.simplify_to)
    pF, _ = igl.polygons_to_triangles(pPI, pPC)

    pnV, pnPI, pnPC = pchs.simplify_convex_hull(hV, hF, args.n)
    pnF, _ = igl.polygons_to_triangles(pnPI, pnPC)
    pchs_vol = triangle_mesh_volume(pnV, pnF)
    print(f"vol(PCHS greedy, n={args.n}) = {pchs_vol:g}")

    search = Search(pV, n=args.n, seed=args.seed, device=args.device)
    t0 = time.perf_counter()
    result = search.run(int(args.trials), trace_every_batches=max(1, int(args.trials) // search.batch_size // 50))
    if str(search.device).startswith("cuda"):
        wp.synchronize_device(search.device)
    dt = time.perf_counter() - t0
    print(
        f"vol(MC search, n={args.n}, {search.trials_run:,} trials) = {result.volume:g}  "
        f"({search.trials_run / dt:,.0f} trials/sec, {dt:.1f}s)"
    )
    print(f"MC / PCHS volume ratio: {result.volume / pchs_vol:.4f}")


if __name__ == "__main__":
    main()
