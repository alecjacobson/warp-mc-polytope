import numpy as np
import pytest
import warp as wp

from mcpolytope.volume import N_MAX, fvecN, polytope_volume_from_halfspaces

wp.init()


@wp.kernel
def _vol_kernel(
    dx: wp.array(dtype=wp.float32, ndim=2),
    dy: wp.array(dtype=wp.float32, ndim=2),
    dz: wp.array(dtype=wp.float32, ndim=2),
    b: wp.array(dtype=wp.float32, ndim=2),
    n: wp.array(dtype=wp.int32),
    big: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    vx = fvecN()
    vy = fvecN()
    vz = fvecN()
    vb = fvecN()
    for k in range(N_MAX):
        vx[k] = dx[i, k]
        vy[k] = dy[i, k]
        vz[k] = dz[i, k]
        vb[k] = b[i, k]
    out[i] = polytope_volume_from_halfspaces(vx, vy, vz, vb, n[i], big[i])


def batch_volumes(dirs_list, b_list, bigs, device="cpu"):
    """Host helper: evaluate polytope_volume_from_halfspaces for a batch of
    (directions, offsets) halfspace sets, one per thread."""
    B = len(dirs_list)
    dx = np.zeros((B, N_MAX), dtype=np.float32)
    dy = np.zeros((B, N_MAX), dtype=np.float32)
    dz = np.zeros((B, N_MAX), dtype=np.float32)
    bb = np.zeros((B, N_MAX), dtype=np.float32)
    ns = np.zeros(B, dtype=np.int32)
    for i, (dirs, b) in enumerate(zip(dirs_list, b_list)):
        n = len(dirs)
        assert n <= N_MAX
        ns[i] = n
        dx[i, :n] = dirs[:, 0]
        dy[i, :n] = dirs[:, 1]
        dz[i, :n] = dirs[:, 2]
        bb[i, :n] = b
    out = wp.zeros(B, dtype=wp.float32, device=device)
    wp.launch(
        _vol_kernel,
        dim=B,
        inputs=[
            wp.array(dx, dtype=wp.float32, device=device),
            wp.array(dy, dtype=wp.float32, device=device),
            wp.array(dz, dtype=wp.float32, device=device),
            wp.array(bb, dtype=wp.float32, device=device),
            wp.array(ns, dtype=wp.int32, device=device),
            wp.array(np.asarray(bigs, dtype=np.float32), dtype=wp.float32, device=device),
            out,
        ],
        device=device,
    )
    wp.synchronize_device(device) if device != "cpu" else None
    return out.numpy()


def volume_of(dirs, b, big, device="cpu"):
    return float(batch_volumes([dirs], [b], [big], device=device)[0])


@pytest.fixture(scope="session")
def gpu_device():
    if not wp.is_device_available("cuda:0"):
        pytest.skip("no CUDA device available")
    return "cuda:0"
