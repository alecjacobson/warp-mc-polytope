"""Experiment (negative result): "lane = facet" tile kernel for the volume step.

One 32-thread tile (block_dim=32) per trial; lane i clips facet i against the
others, reading the other constraints from a shared tile, then tile_sum /
tile_max reduce the volume and invalid flag. Same clipping code as
mcpolytope/volume.py (POLY_MAX = 16).

Result on an L40 (volume step only, directions from an RNG, b from a cheap
formula so no support loop; GPU shared with other jobs, ratios stable):

    n    thread-per-trial   lane=facet tile   ratio
    8        92.4 M/s          50.9 M/s        0.55x
    20       17.2 M/s          10.8 M/s        0.63x
    32        6.84 M/s          5.49 M/s       0.80x

The tile kernel does cut per-thread state a lot (155 -> 106 registers, 1408 ->
384 bytes local), but only n of 32 lanes are active (62% at n=20, 25% at n=8)
and lanes wait on the slowest facet. Packing the constraints into a single vec4
shared read gave identical throughput, so shared-memory extract overhead is not
the limiter. Not integrated into Search. Run:  python experiments/tile_lane_per_facet.py
"""
import sys, time, ctypes
import numpy as np, warp as wp
from mcpolytope.volume import polyvec, _clip_halfplane, _polygon_area, POLY_MAX, _DEGENERATE_REL_TOL, _BOUNDARY_REL_TOL
wp.init()
BD = 32

@wp.kernel(enable_backward=False)
def k_tile(seed: int, n: int, big: float, out: wp.array(dtype=wp.float32)):
    t, lane = wp.tid()
    st = wp.rand_init(seed, t * 32 + lane)
    x = wp.randn(st); y = wp.randn(st); z = wp.randn(st)
    l = wp.sqrt(x * x + y * y + z * z)
    nix = x / l; niy = y / l; niz = z / l
    bi = 0.9 * (1.0 + 0.3 * nix)
    mine = wp.vec4(nix, niy, niz, bi)
    if lane >= n:
        mine = wp.vec4(0.0, 0.0, 0.0, 0.0)
    shared = wp.tile(mine)                      # all lanes' (dir, b) as a tile

    eps = _BOUNDARY_REL_TOL * big
    deg_tol = _DEGENERATE_REL_TOL * big
    contrib = float(0.0)
    bad = float(0.0)
    if lane < n:
        if wp.abs(nix) < 0.9:
            tx = float(1.0); ty = float(0.0); tz = float(0.0)
        else:
            tx = float(0.0); ty = float(1.0); tz = float(0.0)
        ux = niy * tz - niz * ty; uy = niz * tx - nix * tz; uz = nix * ty - niy * tx
        ulen = wp.sqrt(ux * ux + uy * uy + uz * uz)
        ux = ux / ulen; uy = uy / ulen; uz = uz / ulen
        vx = niy * uz - niz * uy; vy = niz * ux - nix * uz; vz = nix * uy - niy * ux
        px = polyvec(); py = polyvec()
        px[0] = -big; py[0] = -big; px[1] = big; py[1] = -big
        px[2] = big; py[2] = big; px[3] = -big; py[3] = big
        count = int(4)
        overflow = bool(False)
        for j in range(n):
            q = wp.vec4(wp.tile_extract(shared, 0, j), wp.tile_extract(shared, 1, j), wp.tile_extract(shared, 2, j), wp.tile_extract(shared, 3, j))
            if j != lane and count > 0:
                a_u = q[0] * ux + q[1] * uy + q[2] * uz
                a_v = q[0] * vx + q[1] * vy + q[2] * vz
                dotij = nix * q[0] + niy * q[1] + niz * q[2]
                rhs = q[3] - bi * dotij
                norm_a = wp.sqrt(a_u * a_u + a_v * a_v)
                if norm_a < deg_tol:
                    if rhs < -deg_tol:
                        count = 0
                    elif dotij > 0.0 and rhs <= deg_tol and j < lane:
                        count = 0
                else:
                    px, py, count, ov = _clip_halfplane(px, py, count, a_u, a_v, rhs)
                    if ov:
                        overflow = True
                        count = 0
        if overflow:
            bad = 1.0
        if count >= 3:
            touches = bool(False)
            for k in range(count):
                if wp.abs(px[k]) > big - eps or wp.abs(py[k]) > big - eps:
                    touches = True
            if touches:
                bad = 1.0
            else:
                contrib = (1.0 / 3.0) * bi * _polygon_area(px, py, count)
    total = wp.tile_sum(wp.tile(contrib))
    anybad = wp.tile_max(wp.tile(bad))
    if lane == 0:
        if anybad[0] > 0.0:
            out[t] = wp.inf
        else:
            out[t] = total[0]

def run(n, N=1 << 20, reps=5):
    dev = wp.get_device("cuda:0")
    out = wp.zeros(N, dtype=wp.float32, device="cuda:0")
    go = lambda: wp.launch_tiled(k_tile, dim=[N], inputs=[0, n, 64.0 * 1.8, out], block_dim=BD, device="cuda:0")
    go(); wp.synchronize_device("cuda:0")
    hooks = k_tile.module.load(dev).get_kernel_hooks(k_tile)
    lib = ctypes.CDLL("libcuda.so.1"); v = ctypes.c_int(0); res = {}
    for nm, a in (("regs", 4), ("local_B", 3), ("shared_B", 1)):
        lib.cuFuncGetAttribute(ctypes.byref(v), a, ctypes.c_void_p(hooks.forward)); res[nm] = v.value
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); go(); wp.synchronize_device("cuda:0"); best = min(best, time.perf_counter() - t0)
    return N / best / 1e6, res, out.numpy()

if __name__ == "__main__":
    for n in (8, 20, 32):
        rate, res, o = run(n)
        print(f"n={n:2d} tile(lane=facet) {rate:6.2f} M trials/s  {res}  finite={np.isfinite(o).mean():.3f}")
