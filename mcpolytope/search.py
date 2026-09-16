import numpy as np
import warp as wp

from mcpolytope.candidates import alias_table, sample_alias, sample_vmf
from mcpolytope.support import support_offset
from mcpolytope.volume import N_MAX, fvecN, polytope_volume_from_halfspaces

ivecN = wp.types.vector(length=N_MAX, dtype=wp.int32)
_MAX_DEDUP_RETRIES = 8


# atomic_min only ever decreases the stored value, so the "not found yet"
# sentinel must be larger than any real trial index, not -1.
_NO_INDEX = np.iinfo(np.int64).max


def _batch_seed(seed, batch_index):
    """Deterministic int32-range seed for one batch, combining the run's
    seed with the batch index. Pure Python (arbitrary precision), used
    identically by run() and regenerate() so trials can be reproduced
    without storing per-trial state.
    """
    return int((int(seed) * 1_000_003 + int(batch_index) + 1) % 2_147_483_647)


@wp.func
def _sample_trial(
    state: wp.uint32,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
):
    """Draw n random unit directions, their support offsets against `points`,
    and the resulting polytope volume. `state` is consumed by value (3*n
    wp.randn draws advance the local copy); the caller doesn't need it back.
    """
    dx = fvecN()
    dy = fvecN()
    dz = fvecN()
    b = fvecN()
    for k in range(n):
        x = wp.randn(state)
        y = wp.randn(state)
        z = wp.randn(state)
        length = wp.sqrt(x * x + y * y + z * z)
        dx[k] = x / length
        dy[k] = y / length
        dz[k] = z / length
    for k in range(n):
        b[k] = support_offset(dx[k], dy[k], dz[k], points, m)
    vol = polytope_volume_from_halfspaces(dx, dy, dz, b, n, big)
    return dx, dy, dz, b, vol


@wp.func
def _sample_trial_candidates(
    state: wp.uint32,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    avoid_duplicates: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
):
    """Like `_sample_trial`, but draws each of the n directions from a
    finite weighted candidate pool (via the alias method) instead of the
    continuous sphere. If avoid_duplicates != 0, retries a bounded number of
    times on an in-trial collision, then falls back to a deterministic
    linear scan for the first still-unused candidate index (guaranteed to
    terminate since Search requires n <= K when avoid_duplicates is set).
    """
    dx = fvecN()
    dy = fvecN()
    dz = fvecN()
    b = fvecN()
    chosen = ivecN()

    for k in range(n):
        state, idx = sample_alias(state, prob, alias, K)

        if avoid_duplicates != 0:
            is_dup = bool(False)
            for j in range(k):
                if chosen[j] == idx:
                    is_dup = True

            # bounded number of resample attempts (no while loops: Warp
            # doesn't like scalar redefinition inside dynamic while loops)
            for _attempt in range(_MAX_DEDUP_RETRIES):
                if is_dup:
                    state, idx = sample_alias(state, prob, alias, K)
                    is_dup = bool(False)
                    for j in range(k):
                        if chosen[j] == idx:
                            is_dup = True

            if is_dup:
                # deterministic fallback: first still-unused candidate index.
                # Terminates because Search enforces n <= K whenever
                # avoid_duplicates is set, so a free index always exists.
                found = bool(False)
                fallback_idx = int(0)
                for cand in range(K):
                    if not found:
                        is_taken = bool(False)
                        for j in range(k):
                            if chosen[j] == cand:
                                is_taken = True
                        if not is_taken:
                            fallback_idx = cand
                            found = True
                idx = fallback_idx

        chosen[k] = idx
        c = candidates[idx]
        dx[k] = c[0]
        dy[k] = c[1]
        dz[k] = c[2]

    for k in range(n):
        b[k] = support_offset(dx[k], dy[k], dz[k], points, m)
    vol = polytope_volume_from_halfspaces(dx, dy, dz, b, n, big)
    return dx, dy, dz, b, vol


@wp.func
def _sample_trial_vmf(
    state: wp.uint32,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    kappa: float,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
):
    """Like `_sample_trial_candidates`, but instead of using the alias-
    picked candidate direction verbatim, treats it as the center of a von
    Mises-Fisher kernel (concentration `kappa`) and draws the actual trial
    direction from that continuous distribution -- so the search can land
    on directions no input face has, while still being guided by the
    weighted candidate signal (kappa=0 ignores it entirely: uniform sphere).
    """
    dx = fvecN()
    dy = fvecN()
    dz = fvecN()
    b = fvecN()

    for k in range(n):
        state, idx = sample_alias(state, prob, alias, K)
        mu = candidates[idx]
        state, d = sample_vmf(state, mu, kappa)
        dx[k] = d[0]
        dy[k] = d[1]
        dz[k] = d[2]

    for k in range(n):
        b[k] = support_offset(dx[k], dy[k], dz[k], points, m)
    vol = polytope_volume_from_halfspaces(dx, dy, dz, b, n, big)
    return dx, dy, dz, b, vol


@wp.func
def _compute_batch_seed(seed: int, batch_index: wp.int64):
    """Device-side twin of `_batch_seed`, used by the graph-capture path
    (where batch_index lives in a device array and can't be baked into the
    graph as a Python constant). Must stay bit-for-bit identical to the
    Python version so `regenerate()` can reproduce any trial.
    """
    modulus = wp.int64(2_147_483_647)
    s64 = wp.int64(seed) * wp.int64(1_000_003) + batch_index + wp.int64(1)
    r = s64 % modulus
    if r < wp.int64(0):
        r = r + modulus
    return wp.int32(r)


@wp.kernel(enable_backward=False)
def _run_trials(
    seed_for_batch: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial(state, points, m, n, big)
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _run_trials_graph(
    seed: int,
    batch_index: wp.array(dtype=wp.int64),
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    """Same as `_run_trials`, but reads the batch index from a device array
    instead of taking seed_for_batch as a launch-time Python constant, so
    the whole batch step can be captured once into a CUDA graph and replayed
    with a different (device-side) batch index each time."""
    tid = wp.tid()
    seed_for_batch = _compute_batch_seed(seed, batch_index[0])
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial(state, points, m, n, big)
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _run_trials_candidates(
    seed_for_batch: int,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    avoid_duplicates: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial_candidates(
        state, candidates, prob, alias, K, avoid_duplicates, points, m, n, big
    )
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _run_trials_candidates_graph(
    seed: int,
    batch_index: wp.array(dtype=wp.int64),
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    avoid_duplicates: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    seed_for_batch = _compute_batch_seed(seed, batch_index[0])
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial_candidates(
        state, candidates, prob, alias, K, avoid_duplicates, points, m, n, big
    )
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _run_trials_vmf(
    seed_for_batch: int,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    kappa: float,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial_vmf(state, candidates, prob, alias, K, kappa, points, m, n, big)
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _run_trials_vmf_graph(
    seed: int,
    batch_index: wp.array(dtype=wp.int64),
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    kappa: float,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    volumes: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    seed_for_batch = _compute_batch_seed(seed, batch_index[0])
    state = wp.rand_init(seed_for_batch, tid)
    _dx, _dy, _dz, _b, vol = _sample_trial_vmf(state, candidates, prob, alias, K, kappa, points, m, n, big)
    volumes[tid] = vol


@wp.kernel(enable_backward=False)
def _reduce_pass1(volumes: wp.array(dtype=wp.float32), global_best_vol: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    wp.atomic_min(global_best_vol, 0, volumes[tid])


@wp.kernel(enable_backward=False)
def _reduce_pass2(
    volumes: wp.array(dtype=wp.float32),
    batch_offset: wp.int64,
    global_best_vol: wp.array(dtype=wp.float32),
    global_best_idx: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    if volumes[tid] == global_best_vol[0]:
        # NOT atomic_min: a later batch's winning trial can have a *larger*
        # global index than an earlier, now-stale best, so the index must be
        # unconditionally claimed (gated on the value match above), not
        # monotonically decreased. Batches run strictly in launch order on
        # one stream, so by the time this runs, global_best_vol already
        # reflects every batch up to and including this one.
        wp.atomic_exch(global_best_idx, 0, batch_offset + wp.int64(tid))


@wp.kernel(enable_backward=False)
def _reduce_pass2_graph(
    volumes: wp.array(dtype=wp.float32),
    batch_size: int,
    batch_index: wp.array(dtype=wp.int64),
    global_best_vol: wp.array(dtype=wp.float32),
    global_best_idx: wp.array(dtype=wp.int64),
):
    tid = wp.tid()
    if volumes[tid] == global_best_vol[0]:
        batch_offset = batch_index[0] * wp.int64(batch_size)
        wp.atomic_exch(global_best_idx, 0, batch_offset + wp.int64(tid))


@wp.kernel(enable_backward=False)
def _advance_batch_index(batch_index: wp.array(dtype=wp.int64)):
    batch_index[0] = batch_index[0] + wp.int64(1)


@wp.kernel(enable_backward=False)
def _regenerate(
    seed_for_batch: int,
    local_idx: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    dirs_out: wp.array(dtype=wp.vec3),
    b_out: wp.array(dtype=wp.float32),
    vol_out: wp.array(dtype=wp.float32),
):
    state = wp.rand_init(seed_for_batch, local_idx)
    dx, dy, dz, b, vol = _sample_trial(state, points, m, n, big)
    for k in range(n):
        dirs_out[k] = wp.vec3(dx[k], dy[k], dz[k])
        b_out[k] = b[k]
    vol_out[0] = vol


@wp.kernel(enable_backward=False)
def _regenerate_candidates(
    seed_for_batch: int,
    local_idx: int,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    avoid_duplicates: int,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    dirs_out: wp.array(dtype=wp.vec3),
    b_out: wp.array(dtype=wp.float32),
    vol_out: wp.array(dtype=wp.float32),
):
    state = wp.rand_init(seed_for_batch, local_idx)
    dx, dy, dz, b, vol = _sample_trial_candidates(
        state, candidates, prob, alias, K, avoid_duplicates, points, m, n, big
    )
    for k in range(n):
        dirs_out[k] = wp.vec3(dx[k], dy[k], dz[k])
        b_out[k] = b[k]
    vol_out[0] = vol


@wp.kernel(enable_backward=False)
def _regenerate_vmf(
    seed_for_batch: int,
    local_idx: int,
    candidates: wp.array(dtype=wp.vec3),
    prob: wp.array(dtype=wp.float32),
    alias: wp.array(dtype=wp.int32),
    K: int,
    kappa: float,
    points: wp.array(dtype=wp.vec3),
    m: int,
    n: int,
    big: float,
    dirs_out: wp.array(dtype=wp.vec3),
    b_out: wp.array(dtype=wp.float32),
    vol_out: wp.array(dtype=wp.float32),
):
    state = wp.rand_init(seed_for_batch, local_idx)
    dx, dy, dz, b, vol = _sample_trial_vmf(state, candidates, prob, alias, K, kappa, points, m, n, big)
    for k in range(n):
        dirs_out[k] = wp.vec3(dx[k], dy[k], dz[k])
        b_out[k] = b[k]
    vol_out[0] = vol


class SearchResult:
    def __init__(self, volume, index, seed, directions, offsets, trace):
        self.volume = volume
        self.index = index
        self.seed = seed
        self.directions = directions
        self.offsets = offsets
        self.trace = trace

    def __repr__(self):
        return f"SearchResult(volume={self.volume!r}, index={self.index!r}, seed={self.seed!r})"


class Search:
    """GPU Monte Carlo search for the minimum-volume n-halfspace polytope
    enclosing `points`, entirely in Warp (no per-trial host sync).

    Each trial samples n random unit directions and builds halfspaces from
    their support values against `points` (so every trial's polytope always
    contains `points`), then computes its exact volume on-device via
    `polytope_volume_from_halfspaces`. `batch_size` trials run per kernel
    launch; batches are seeded deterministically from (seed, batch_index) so
    the winning trial's directions can be regenerated after the fact
    (`regenerate`/`best`) instead of storing every trial's directions.
    """

    def __init__(
        self,
        points,
        n,
        seed=0,
        big_factor=64.0,
        batch_size=1 << 20,
        device=None,
        candidates=None,
        weights=None,
        avoid_duplicates=False,
        kappa=None,
    ):
        if n < 4 or n > N_MAX:
            raise ValueError(f"n must be in [4, {N_MAX}], got {n}")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (m, 3)")

        self.centroid = points.mean(axis=0)
        recentered = points - self.centroid
        self.R = float(np.max(np.linalg.norm(recentered, axis=1)))
        if self.R <= 0:
            raise ValueError("degenerate point set (zero radius after centering)")
        self.big = big_factor * self.R

        self.n = n
        self.seed = seed
        self.batch_size = int(batch_size)
        self.device = device if device is not None else wp.get_preferred_device()
        self.m = recentered.shape[0]
        self.points = wp.array(recentered.astype(np.float32), dtype=wp.vec3, device=self.device)

        self.candidates = None
        self.avoid_duplicates = bool(avoid_duplicates)
        if candidates is not None:
            candidates = np.asarray(candidates, dtype=np.float64)
            if candidates.ndim != 2 or candidates.shape[1] != 3:
                raise ValueError("candidates must have shape (K, 3)")
            K = candidates.shape[0]
            if K == 0:
                raise ValueError("candidates must be non-empty")
            if self.avoid_duplicates and n > K:
                raise ValueError(f"avoid_duplicates=True requires n <= K (n={n}, K={K})")
            unit_candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

            self.K = K
            self.candidates = wp.array(unit_candidates.astype(np.float32), dtype=wp.vec3, device=self.device)
            weights = np.ones(K) if weights is None else np.asarray(weights, dtype=np.float64)
            prob_np, alias_np = alias_table(weights)
            self.prob = wp.array(prob_np, dtype=wp.float32, device=self.device)
            self.alias = wp.array(alias_np, dtype=wp.int32, device=self.device)
        elif weights is not None:
            raise ValueError("weights requires candidates to be given")

        self.kappa = None
        if kappa is not None:
            if self.candidates is None:
                raise ValueError("kappa requires candidates to be given")
            if self.avoid_duplicates:
                raise ValueError(
                    "avoid_duplicates is not supported with kappa (continuous vMF draws "
                    "have zero probability of exact duplicates; a without-replacement "
                    "scheme isn't built for this mode)"
                )
            self.kappa = float(kappa)

        self.global_best_vol = wp.array(np.array([np.inf], dtype=np.float32), dtype=wp.float32, device=self.device)
        self.global_best_idx = wp.array(np.array([_NO_INDEX], dtype=np.int64), dtype=wp.int64, device=self.device)

        self.trials_run = 0
        self.trace = []  # list of (trials_run, best_volume)

        self._device_batch_index = wp.array(np.array([0], dtype=np.int64), dtype=wp.int64, device=self.device)
        self._graph = None
        self._graph_volumes = None

    def run(self, num_trials, trace_every_batches=None, use_graph=False):
        """Run `num_trials` additional Monte Carlo trials (in batches of
        `self.batch_size`). If `trace_every_batches` is set, records
        (trials_run, best_volume_so_far) that often (each checkpoint costs
        one small device->host scalar copy; no other host syncs occur).

        `use_graph=True` captures the whole "one batch" step (trial kernel,
        reduction, batch-index advance) as a single CUDA graph the first
        time it's needed and replays it thereafter, amortizing per-launch
        CPU overhead across `num_trials`. Only available on CUDA devices;
        must not be mixed with `trace_every_batches` finer than the whole
        call (trace checkpoints force a graph break to read back the host
        scalar).
        """
        if self.trials_run % self.batch_size != 0:
            raise RuntimeError("trials_run is not batch-aligned; use a consistent batch_size")
        if use_graph and not str(self.device).startswith("cuda"):
            raise ValueError("use_graph=True requires a CUDA device")

        remaining = int(num_trials)
        batches_done = 0

        if use_graph:
            # keep the device-side batch counter in sync in case earlier
            # (ungraphed) batches ran since it was last updated.
            self._device_batch_index.assign(np.array([self.trials_run // self.batch_size], dtype=np.int64))
            if self._graph_volumes is None or self._graph_volumes.shape[0] != self.batch_size:
                self._graph_volumes = wp.empty(self.batch_size, dtype=wp.float32, device=self.device)
            if self._graph is None:
                with wp.ScopedCapture(device=self.device) as capture:
                    if self.kappa is not None:
                        self._one_batch_step_graph_vmf(self._graph_volumes)
                    elif self.candidates is not None:
                        self._one_batch_step_graph_candidates(self._graph_volumes)
                    else:
                        self._one_batch_step_graph(self._graph_volumes)
                self._graph = capture.graph

            while remaining > 0:
                B = min(self.batch_size, remaining)
                if B != self.batch_size:
                    # partial final batch: fall back to the ungraphed path
                    # (graph body is sized for a full batch)
                    break
                wp.capture_launch(self._graph)
                self.trials_run += B
                remaining -= B
                batches_done += 1
                if trace_every_batches and batches_done % trace_every_batches == 0:
                    self.trace.append((self.trials_run, float(self.global_best_vol.numpy()[0])))

        volumes = wp.empty(self.batch_size, dtype=wp.float32, device=self.device) if remaining > 0 else None
        while remaining > 0:
            B = min(self.batch_size, remaining)
            batch_index = self.trials_run // self.batch_size
            seed_for_batch = _batch_seed(self.seed, batch_index)

            if self.kappa is not None:
                wp.launch(
                    _run_trials_vmf,
                    dim=B,
                    inputs=[
                        seed_for_batch,
                        self.candidates,
                        self.prob,
                        self.alias,
                        self.K,
                        self.kappa,
                        self.points,
                        self.m,
                        self.n,
                        self.big,
                        volumes,
                    ],
                    device=self.device,
                )
            elif self.candidates is not None:
                wp.launch(
                    _run_trials_candidates,
                    dim=B,
                    inputs=[
                        seed_for_batch,
                        self.candidates,
                        self.prob,
                        self.alias,
                        self.K,
                        int(self.avoid_duplicates),
                        self.points,
                        self.m,
                        self.n,
                        self.big,
                        volumes,
                    ],
                    device=self.device,
                )
            else:
                wp.launch(
                    _run_trials,
                    dim=B,
                    inputs=[seed_for_batch, self.points, self.m, self.n, self.big, volumes],
                    device=self.device,
                )
            wp.launch(_reduce_pass1, dim=B, inputs=[volumes, self.global_best_vol], device=self.device)
            wp.launch(
                _reduce_pass2,
                dim=B,
                inputs=[volumes, wp.int64(self.trials_run), self.global_best_vol, self.global_best_idx],
                device=self.device,
            )

            self.trials_run += B
            remaining -= B
            batches_done += 1

            if trace_every_batches and batches_done % trace_every_batches == 0:
                self.trace.append((self.trials_run, float(self.global_best_vol.numpy()[0])))

        return self.best()

    def _one_batch_step_graph(self, volumes):
        B = self.batch_size
        wp.launch(
            _run_trials_graph,
            dim=B,
            inputs=[self.seed, self._device_batch_index, self.points, self.m, self.n, self.big, volumes],
            device=self.device,
        )
        wp.launch(_reduce_pass1, dim=B, inputs=[volumes, self.global_best_vol], device=self.device)
        wp.launch(
            _reduce_pass2_graph,
            dim=B,
            inputs=[volumes, B, self._device_batch_index, self.global_best_vol, self.global_best_idx],
            device=self.device,
        )
        wp.launch(_advance_batch_index, dim=1, inputs=[self._device_batch_index], device=self.device)

    def _one_batch_step_graph_candidates(self, volumes):
        B = self.batch_size
        wp.launch(
            _run_trials_candidates_graph,
            dim=B,
            inputs=[
                self.seed,
                self._device_batch_index,
                self.candidates,
                self.prob,
                self.alias,
                self.K,
                int(self.avoid_duplicates),
                self.points,
                self.m,
                self.n,
                self.big,
                volumes,
            ],
            device=self.device,
        )
        wp.launch(_reduce_pass1, dim=B, inputs=[volumes, self.global_best_vol], device=self.device)
        wp.launch(
            _reduce_pass2_graph,
            dim=B,
            inputs=[volumes, B, self._device_batch_index, self.global_best_vol, self.global_best_idx],
            device=self.device,
        )
        wp.launch(_advance_batch_index, dim=1, inputs=[self._device_batch_index], device=self.device)

    def _one_batch_step_graph_vmf(self, volumes):
        B = self.batch_size
        wp.launch(
            _run_trials_vmf_graph,
            dim=B,
            inputs=[
                self.seed,
                self._device_batch_index,
                self.candidates,
                self.prob,
                self.alias,
                self.K,
                self.kappa,
                self.points,
                self.m,
                self.n,
                self.big,
                volumes,
            ],
            device=self.device,
        )
        wp.launch(_reduce_pass1, dim=B, inputs=[volumes, self.global_best_vol], device=self.device)
        wp.launch(
            _reduce_pass2_graph,
            dim=B,
            inputs=[volumes, B, self._device_batch_index, self.global_best_vol, self.global_best_idx],
            device=self.device,
        )
        wp.launch(_advance_batch_index, dim=1, inputs=[self._device_batch_index], device=self.device)

    def regenerate(self, gidx):
        batch_index, local_idx = divmod(int(gidx), self.batch_size)
        seed_for_batch = _batch_seed(self.seed, batch_index)
        dirs_out = wp.zeros(self.n, dtype=wp.vec3, device=self.device)
        b_out = wp.zeros(self.n, dtype=wp.float32, device=self.device)
        vol_out = wp.zeros(1, dtype=wp.float32, device=self.device)
        if self.kappa is not None:
            wp.launch(
                _regenerate_vmf,
                dim=1,
                inputs=[
                    seed_for_batch,
                    local_idx,
                    self.candidates,
                    self.prob,
                    self.alias,
                    self.K,
                    self.kappa,
                    self.points,
                    self.m,
                    self.n,
                    self.big,
                    dirs_out,
                    b_out,
                    vol_out,
                ],
                device=self.device,
            )
        elif self.candidates is not None:
            wp.launch(
                _regenerate_candidates,
                dim=1,
                inputs=[
                    seed_for_batch,
                    local_idx,
                    self.candidates,
                    self.prob,
                    self.alias,
                    self.K,
                    int(self.avoid_duplicates),
                    self.points,
                    self.m,
                    self.n,
                    self.big,
                    dirs_out,
                    b_out,
                    vol_out,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _regenerate,
                dim=1,
                inputs=[seed_for_batch, local_idx, self.points, self.m, self.n, self.big, dirs_out, b_out, vol_out],
                device=self.device,
            )
        return dirs_out.numpy(), b_out.numpy(), float(vol_out.numpy()[0])

    def best(self):
        vol = float(self.global_best_vol.numpy()[0])
        idx = int(self.global_best_idx.numpy()[0])
        directions = offsets = None
        if idx != _NO_INDEX:
            directions, offsets, regen_vol = self.regenerate(idx)
            if not np.isclose(regen_vol, vol, rtol=1e-4):
                raise RuntimeError(
                    f"regenerated volume {regen_vol} does not match recorded best {vol} "
                    f"for index {idx} -- RNG/seed mismatch between run() and regenerate()"
                )
            # directions/offsets are in the recentered frame; shift offsets
            # back so the halfspaces apply to the *original* point cloud.
            offsets = offsets + directions @ self.centroid
        return SearchResult(vol, idx, self.seed, directions, offsets, list(self.trace))
