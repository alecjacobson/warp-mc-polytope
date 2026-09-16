"""Compare candidate-direction weighting schemes (and avoid_duplicates) for
the GPU Monte Carlo polytope search against PCHS's greedy simplification.

Usage:
    python examples/weight_experiment.py path/to/Actaeon.ply --n 20 --trials 1e7
"""

import argparse
import time

import igl
import igl.copyleft.cgal
import numpy as np
import pchs
import warp as wp

from mcpolytope.search import Search
from mcpolytope.weights import WEIGHT_FUNCS, hull_face_candidates


def triangle_mesh_volume(V, F):
    m0, _, _ = igl.moments(V, F)
    return m0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("-n", type=int, default=20)
    parser.add_argument("--trials", type=float, default=1e7)
    parser.add_argument("--simplify-to", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    wp.init()

    V, F = igl.read_triangle_mesh(args.mesh)
    hF = igl.copyleft.cgal.convex_hull(V)
    hV, hF, _, _ = igl.remove_unreferenced(V, hF)

    pV, pPI, pPC = pchs.simplify_convex_hull(hV, hF, args.simplify_to)

    pnV, pnPI, pnPC = pchs.simplify_convex_hull(hV, hF, args.n)
    pnF, _ = igl.polygons_to_triangles(pnPI, pnPC)
    pchs_vol = triangle_mesh_volume(pnV, pnF)
    print(f"PCHS greedy (n={args.n}): {pchs_vol:g}")
    print(f"{'scheme':<14} {'avoid_dup':<10} {'volume':>12} {'/PCHS':>8} {'trials/sec':>12}")

    normals, areas = hull_face_candidates(hV, hF)

    def run(candidates, weights, avoid_duplicates, seed):
        s = Search(
            pV,
            n=args.n,
            seed=seed,
            candidates=candidates,
            weights=weights,
            avoid_duplicates=avoid_duplicates,
            device=args.device,
        )
        t0 = time.perf_counter()
        res = s.run(int(args.trials))
        if str(s.device).startswith("cuda"):
            wp.synchronize_device(s.device)
        dt = time.perf_counter() - t0
        return res.volume, s.trials_run / dt

    # random-sphere baseline (no candidates)
    s = Search(pV, n=args.n, seed=args.seed, device=args.device)
    t0 = time.perf_counter()
    res = s.run(int(args.trials))
    if str(s.device).startswith("cuda"):
        wp.synchronize_device(s.device)
    dt = time.perf_counter() - t0
    print(f"{'random-sphere':<14} {'n/a':<10} {res.volume:>12.5f} {res.volume / pchs_vol:>8.4f} {s.trials_run / dt:>12,.0f}")

    for name, weight_fn in WEIGHT_FUNCS.items():
        weights = weight_fn(hV, hF, normals, areas)
        for avoid_duplicates in (False, True):
            vol, rate = run(normals, weights, avoid_duplicates, args.seed)
            print(f"{name:<14} {str(avoid_duplicates):<10} {vol:>12.5f} {vol / pchs_vol:>8.4f} {rate:>12,.0f}")


if __name__ == "__main__":
    main()
