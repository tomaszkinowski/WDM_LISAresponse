r"""
`fast_split` on a GPU: the same stages, the two lag sums as Pallas kernels.

- `shared_sums_kernel`: the amplitude-rate and sinc lag sums of both
  polarisations. Their weights are constants of (l, sigma), so they are baked
  into the kernel. Which terms carry :math:`(-1)^m` is fixed by sigma
  (sigma = +-1 always, sigma = 0 never) and which carry :math:`(-1)^n` by the
  parity of :math:`l+\sigma`, so the kernel keeps one accumulator per (sum,
  sigma, lag parity) and reads each input column once per lag; the
  :math:`m+\sigma` column offset and the signs are applied afterwards
  (`_recombine_sums`), as in production's GPU shift.
- `rate_shift_kernel`: the rate-corrected shift from the per-row weights of
  `fast_split.shift_weights` (12 per lag and row: :math:`p_c, p_s, q_c, q_s`
  per sigma), one accumulator per (sigma, cos/sin), the :math:`m\Delta F`
  factor formed in the kernel from the output column. Column offset, signs
  and the carrier follow in `_recombine_shift`.

Kernels and recombinations run in float32; geometry, delays and the per-row
weights are formed in float64 and cast.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

import fast_split as fs

BLOCK_ROWS, BLOCK_COLS, NUM_WARPS = 8, 256, 4
INTERPRET = False      # run the kernels in the Pallas interpreter (testing)
#: rate kernel launch, A400 at L=25: one program per sigma, 4x256 tiles, 4 warps
#: (Nf=256 16.5 ms, Nf=2500 17.2 ms per link; first version 56.5 ms)
RATE_TILE, RATE_WARPS = (4, 256), 4
#: apply the carrier inside the rate kernel (one grid per sigma, not two):
#: libdevice cos/sin in the kernel made it 572 ms against 20 (A400, Nf=256)
CARRIER_IN_KERNEL = False


def _tiles(Nt, Nf):
    cols = 16
    while cols < min(Nf, BLOCK_COLS):
        cols *= 2
    rows = max(BLOCK_ROWS, 1024//cols)
    return rows, cols, -(-Nt//rows)*rows, -(-Nf//cols)*cols


def _extend(grids, L, Nt_pad, Nf_pad):
    """Cyclic extension by L rows each end, zero padding to whole tiles."""
    G, Nt, Nf = grids.shape
    ext = jnp.concatenate([grids[:, -L:], grids, grids[:, :L]], axis=1)
    return jnp.pad(ext, ((0, 0), (0, Nt_pad - Nt), (0, Nf_pad - Nf)))


# ---------------------------------------------------------------- shared sums

def _sum_terms(amp_w, sinc_w):
    r"""
    The accumulators of the shared sums: a list of (sum, sigma, parity) and,
    per lag index i, the (accumulator, weight) pairs it feeds.
    """
    L = (amp_w.shape[1] - 1)//2
    keys = [("amp", -1, 0), ("amp", -1, 1), ("amp", 1, 0), ("amp", 1, 1),
            ("sinc", -1, 0), ("sinc", -1, 1), ("sinc", 1, 0), ("sinc", 1, 1),
            ("sinc", 0, 1)]
    feeds = []
    for i in range(2*L + 1):
        p = (i - L) % 2
        row = []
        for a, (name, sigma, parity) in enumerate(keys):
            if parity == p:
                w = (amp_w if name == "amp" else sinc_w)[sigma + 1, i]
                row.append((a, float(w)))
        feeds.append(row)
    return keys, feeds


def shared_sums_kernel(amp_w, sinc_w, grids):
    """The raw accumulators, shape (G, 9, Nt, Nf), for grids (G, Nt, Nf)."""
    keys, feeds = _sum_terms(np.asarray(amp_w), np.asarray(sinc_w))
    L = len(feeds)//2
    G, Nt, Nf = grids.shape
    rows, cols, Nt_pad, Nf_pad = _tiles(Nt, Nf)
    ext = _extend(grids, L, Nt_pad, Nf_pad)
    dtype = grids.dtype

    def kernel(ext_ref, out_ref):
        g = pl.program_id(0)
        i0 = pl.program_id(1)*rows
        j0 = pl.program_id(2)*cols
        acc = [jnp.zeros((rows, cols), dtype) for _ in keys]
        for i, row in enumerate(feeds):
            l = i - L
            x = ext_ref[g, pl.ds(i0 + L - l, rows), pl.ds(j0, cols)]
            for a, w in row:
                acc[a] += w*x
        for a in range(len(keys)):
            out_ref[g, a, pl.ds(i0, rows), pl.ds(j0, cols)] = acc[a]

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((G, len(keys), Nt_pad, Nf_pad), dtype),
        grid=(G, Nt_pad//rows, Nf_pad//cols),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=1),
        interpret=INTERPRET,
    )(ext)
    return keys, out[:, :, :Nt, :Nf]


def _recombine_sums(keys, acc):
    r"""
    ``amp`` (Nt, Nf) and ``sinc`` (3 grids, sigma = -1, 0, 1) of one grid from
    its accumulators: :math:`(-1)^n` on the parity where :math:`l+\sigma` is
    odd, column :math:`m+\sigma`, then :math:`(-1)^m` for sigma = +-1.
    """
    _, Nt, Nf = acc.shape
    sign_n = jnp.where(jnp.arange(Nt) % 2 == 0, 1.0, -1.0).astype(acc.dtype)[:, None]
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(acc.dtype)[None, :]
    parts = {("amp", -1): 0.0, ("amp", 1): 0.0,
             ("sinc", -1): 0.0, ("sinc", 0): 0.0, ("sinc", 1): 0.0}
    for a, (name, sigma, parity) in enumerate(keys):
        x = acc[a]*sign_n if (parity + sigma) % 2 else acc[a]
        parts[(name, sigma)] = parts[(name, sigma)] + x
    move = lambda x, sigma: x if sigma == 0 else jnp.roll(x, -sigma, axis=1)*sign_m
    amp = move(parts[("amp", -1)], -1) + move(parts[("amp", 1)], 1)
    sinc = tuple(move(parts[("sinc", s)], s) for s in (-1, 0, 1))
    return amp, sinc


OUT5 = (("amp", -1), ("amp", 1), ("sinc", -1), ("sinc", 0), ("sinc", 1))


def shared_sums_kernel5(amp_w, sinc_w, grids):
    r"""
    `shared_sums_kernel` with the two lag parities of each (sum, sigma)
    joined in the kernel, the :math:`(-1)^n` of odd :math:`l+\sigma` taken
    from the row index: five grids per input, ordered as `OUT5`, still
    before the :math:`m+\sigma` column offset. Shape (G, 5, Nt, Nf).
    """
    keys, feeds = _sum_terms(np.asarray(amp_w), np.asarray(sinc_w))
    L = len(feeds)//2
    G, Nt, Nf = grids.shape
    rows, cols, Nt_pad, Nf_pad = _tiles(Nt, Nf)
    ext = _extend(grids, L, Nt_pad, Nf_pad)
    dtype = grids.dtype

    def kernel(ext_ref, out_ref):
        g = pl.program_id(0)
        i0 = pl.program_id(1)*rows
        j0 = pl.program_id(2)*cols
        acc = [jnp.zeros((rows, cols), dtype) for _ in keys]
        for i, row in enumerate(feeds):
            l = i - L
            x = ext_ref[g, pl.ds(i0 + L - l, rows), pl.ds(j0, cols)]
            for a, w in row:
                acc[a] += w*x
        n = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0) + i0
        sign_n = jnp.where(n % 2 == 0, 1.0, -1.0).astype(dtype)
        for o, (name, sigma) in enumerate(OUT5):
            total = 0.0
            for a, (nm, sg, parity) in enumerate(keys):
                if (nm, sg) == (name, sigma):
                    total = total + (acc[a]*sign_n if (parity + sigma) % 2 else acc[a])
            out_ref[g, o, pl.ds(i0, rows), pl.ds(j0, cols)] = total

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((G, len(OUT5), Nt_pad, Nf_pad), dtype),
        grid=(G, Nt_pad//rows, Nf_pad//cols),
        compiler_params=plgpu.CompilerParams(num_warps=NUM_WARPS, num_stages=1),
        interpret=INTERPRET,
    )(ext)
    return out[:, :, :Nt, :Nf]


def _recombine_sums5(acc):
    """``amp`` and ``sinc`` of one grid from `shared_sums_kernel5`'s five."""
    Nf = acc.shape[-1]
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(acc.dtype)[None, :]
    move = lambda x, sigma: x if sigma == 0 else jnp.roll(x, -sigma, axis=1)*sign_m
    amp = move(acc[0], -1) + move(acc[1], 1)
    return amp, (move(acc[2], -1), acc[3], move(acc[4], 1))


# ---------------------------------------------------------------- rate shift

def rate_shift_kernel(weights, grids, dF, by_sigma=False, tile=None, warps=None,
                      cycles=None):
    r"""
    :math:`K^\sigma_{c,s}[n, m'] = \sum_l \omega_{(n-l)m'}\,(p^{l\sigma}_{c,s}(n)
    + m\Delta F\, q^{l\sigma}_{c,s}(n))`, with :math:`m=(m'-\sigma) \bmod N_f`
    the output column, shape (G, 6, Nt, Nf) ordered (sigma, cos/sin).

    weights : (G, 2L+1, 3, 4, Nt), from `fast_split.shift_weights`.
    by_sigma : one program per sigma (2 accumulators, 4 weight loads per lag)
        instead of all three in one (6 and 12).
    tile, warps : (rows, cols) and warps per program, for tuning.
    cycles : optional (G, Nt) float32, :math:`\Delta F\delta_n`. With it (and
        ``by_sigma``) each program applies the carrier
        :math:`\cos, \sin(2\pi m\Delta F\delta_n)` itself and writes one grid
        per sigma: (G, 3, Nt, Nf), before the column offset. Small phases
        only (the split model's rest delay: a cycle or two), so plain
        float32 trig.
    """
    G, Nt, Nf = grids.shape
    L = (weights.shape[1] - 1)//2
    rows, cols, Nt_pad, Nf_pad = _tiles(Nt, Nf)
    if tile is not None:
        rows, cols = tile
        Nt_pad, Nf_pad = -(-Nt//rows)*rows, -(-Nf//cols)*cols
    dtype = grids.dtype
    ext = _extend(grids, L, Nt_pad, Nf_pad)
    w = weights.reshape(G, 2*L + 1, 12, Nt).astype(dtype)
    w = jnp.pad(w, ((0, 0), (0, 0), (0, 0), (0, Nt_pad - Nt)))
    sigmas = (None,) if not by_sigma else (0, 1, 2)
    epilogue = cycles is not None
    assert by_sigma or not epilogue
    inputs = (ext, w)
    if epilogue:
        cyc = jnp.pad(cycles.astype(jnp.float32), ((0, 0), (0, Nt_pad - Nt)))
        inputs = (ext, w, cyc)

    def kernel(ext_ref, w_ref, *refs):
        out_ref = refs[-1]
        g = pl.program_id(0)
        i0 = pl.program_id(1)*rows
        j0 = pl.program_id(2)*cols
        ks = range(3) if not by_sigma else (pl.program_id(3),)
        col = jax.lax.broadcasted_iota(jnp.int32, (1, cols), 1) + j0
        m_dF = []
        for k in ks:
            m = col - (k - 1)
            m = jnp.where(m < 0, m + Nf, jnp.where(m >= Nf, m - Nf, m))
            m_dF.append(m.astype(dtype)*dF)
        acc = [jnp.zeros((rows, cols), dtype) for _ in range(2*len(m_dF))]
        for i in range(2*L + 1):
            l = i - L
            x = ext_ref[g, pl.ds(i0 + L - l, rows), pl.ds(j0, cols)]
            for a, k in enumerate(ks):
                p_c, p_s, q_c, q_s = (w_ref[g, i, 4*k + c, pl.ds(i0, rows)][:, None]
                                      for c in range(4))
                acc[2*a] += (p_c + m_dF[a]*q_c)*x
                acc[2*a + 1] += (p_s + m_dF[a]*q_s)*x
        if epilogue:
            k = ks[0]
            m = col - (k - 1)
            m = jnp.where(m < 0, m + Nf, jnp.where(m >= Nf, m - Nf, m))
            c = refs[0][g, pl.ds(i0, rows)][:, None]
            theta = 2*jnp.pi*(m.astype(jnp.float32)*c)
            out = (jnp.cos(theta)*acc[0].astype(jnp.float32)
                   + jnp.sin(theta)*acc[1].astype(jnp.float32))
            out_ref[g, k, pl.ds(i0, rows), pl.ds(j0, cols)] = out
            return
        for a, k in enumerate(ks):
            out_ref[g, 2*k, pl.ds(i0, rows), pl.ds(j0, cols)] = acc[2*a]
            out_ref[g, 2*k + 1, pl.ds(i0, rows), pl.ds(j0, cols)] = acc[2*a + 1]

    grid = (G, Nt_pad//rows, Nf_pad//cols) + ((3,) if by_sigma else ())
    shape = (G, 3, Nt_pad, Nf_pad) if epilogue else (G, 6, Nt_pad, Nf_pad)
    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32 if epilogue else dtype),
        grid=grid,
        compiler_params=plgpu.CompilerParams(num_warps=warps or NUM_WARPS, num_stages=1),
        interpret=INTERPRET,
    )(*inputs)
    return out[:, :, :Nt, :Nf]


def _recombine_carried(K):
    """The shifted grid from the three carried sums of `rate_shift_kernel`."""
    Nf = K.shape[-1]
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(K.dtype)[None, :]
    return (jnp.roll(K[0], 1, axis=1)*sign_m + K[1]
            + jnp.roll(K[2], -1, axis=1)*sign_m)


def rate_shift_kernel_poly(rowvals, grids, dF, coef, tile=None, warps=None):
    r"""
    `rate_shift_kernel` with every filter formed in the kernel from the
    polynomials of `fast_split.poly_filters` (coefficients baked in), so no
    per-row weights are read: each program loads its rows' scaled delay,
    rate and half-bin cos/sin once and evaluates the filters and weights of
    each lag on those rows. One call per sigma.

    rowvals : (G, 4, Nt) float32: :math:`-\delta_n/B`, :math:`\dot\delta_n`,
        :math:`\cos`, :math:`\sin(\pi\Delta F\delta_n)`.
    coef : numpy (14, 2L+1, K+1), descending powers.
    Returns (G, 6, Nt, Nf), ordered (sigma, cos/sin) as `rate_shift_kernel`.
    """
    G, Nt, Nf = grids.shape
    L = (coef.shape[1] - 1)//2
    rows, cols, Nt_pad, Nf_pad = _tiles(Nt, Nf)
    if tile is not None:
        rows, cols = tile
        Nt_pad, Nf_pad = -(-Nt//rows)*rows, -(-Nf//cols)*cols
    dtype = grids.dtype
    ext = _extend(grids, L, Nt_pad, Nf_pad)
    rv = jnp.pad(rowvals.astype(dtype), ((0, 0), (0, 0), (0, Nt_pad - Nt)))
    scale = np.abs(coef).max(axis=(1, 2))
    live = scale > 1e-12*scale.max()                 # sigma=0 U_r, V_i vanish
    coef = [[[float(c) for c in coef[f, i]] for i in range(2*L + 1)]
            for f in range(coef.shape[0])]
    i_pow = ((1., 0.), (0., 1.), (-1., 0.), (0., -1.))

    def call(k):
        sigma = k - 1

        def kernel(ext_ref, rv_ref, out_ref):
            g = pl.program_id(0)
            i0 = pl.program_id(1)*rows
            j0 = pl.program_id(2)*cols
            r = pl.ds(i0, rows)
            s, rate = rv_ref[g, 0, r], rv_ref[g, 1, r]
            cos_h, sin_h = rv_ref[g, 2, r], rv_ref[g, 3, r]
            n = jax.lax.broadcasted_iota(jnp.int32, (rows,), 0) + i0
            sign_n = jnp.where(n % 2 == 0, 1.0, -1.0).astype(dtype)
            col = jax.lax.broadcasted_iota(jnp.int32, (1, cols), 1) + j0
            m = col - sigma
            m = jnp.where(m < 0, m + Nf, jnp.where(m >= Nf, m - Nf, m))
            m_dF = m.astype(dtype)*dF
            cos_s = 1.0 if sigma == 0 else cos_h
            sin_s = 0.0 if sigma == 0 else sigma*sin_h

            def poly(f, i):
                if not live[f]:
                    return 0.0
                c = coef[f][i]
                out = c[0] + 0.0*s
                for ck in c[1:]:
                    out = out*s + ck
                return out

            acc_c = jnp.zeros((rows, cols), dtype)
            acc_s = jnp.zeros((rows, cols), dtype)
            for i in range(2*L + 1):
                l = i - L
                i_re, i_im = i_pow[l % 4]
                lag_re, lag_im = ((1.0, 0.0) if sigma == 0 else
                                  (i_re, i_im if sigma == -1 else -i_im))
                a = lag_re*cos_s + lag_im*sin_s
                b = lag_im*cos_s - lag_re*sin_s
                if (l + sigma) % 2 != 0:
                    a, b = -sign_n*b, sign_n*a
                T = poly(0 if sigma == 0 else 1, i)
                Ur, Ui, Vr, Vi = (poly(2 + 4*k + c, i)*rate for c in range(4))
                q_c, q_s = a*Ur + b*Ui, b*Ur - a*Ui
                shift = 0.5*sigma*dF
                p_c = a*(T + Vr) + b*Vi + shift*q_c
                p_s = b*(T + Vr) - a*Vi + shift*q_s
                x = ext_ref[g, pl.ds(i0 + L - l, rows), pl.ds(j0, cols)]
                acc_c += (p_c[:, None] + m_dF*q_c[:, None])*x
                acc_s += (p_s[:, None] + m_dF*q_s[:, None])*x
            out_ref[g, 0, r, pl.ds(j0, cols)] = acc_c
            out_ref[g, 1, r, pl.ds(j0, cols)] = acc_s

        return pl.pallas_call(
            kernel, out_shape=jax.ShapeDtypeStruct((G, 2, Nt_pad, Nf_pad), dtype),
            grid=(G, Nt_pad//rows, Nf_pad//cols),
            compiler_params=plgpu.CompilerParams(num_warps=warps or NUM_WARPS,
                                                 num_stages=1),
            interpret=INTERPRET,
        )(ext, rv)

    out = jnp.concatenate([call(k) for k in range(3)], axis=1)
    return out[:, :, :Nt, :Nf]


def poly_rowvals(rest, dT, dF, bound=None):
    r"""The per-row inputs of `rate_shift_kernel_poly` for one link, (4, Nt)."""
    bound = fs.REST_BOUND if bound is None else bound
    half = jnp.pi*dF*rest
    return jnp.stack([-rest/bound, jnp.gradient(rest, dT), jnp.cos(half), jnp.sin(half)])


def _recombine_shift(K, delta, dF):
    """The shifted grid from `rate_shift_kernel`'s six sums, (Nt, Nf)."""
    from WDM.code.time_delay_filters.shift_gpu import carrier

    _, Nt, Nf = K.shape
    sign_m = jnp.where(jnp.arange(Nf) % 2 == 0, 1.0, -1.0).astype(K.dtype)[None, :]
    cos_part, sin_part = 0.0, 0.0
    for k, sigma in enumerate((-1, 0, 1)):
        c, s = K[2*k], K[2*k + 1]
        if sigma != 0:
            c = jnp.roll(c, -sigma, axis=1)*sign_m
            s = jnp.roll(s, -sigma, axis=1)*sign_m
        cos_part, sin_part = cos_part + c, sin_part + s
    cos_t, sin_t = carrier(dF, delta, Nf, K.dtype)
    return cos_t*cos_part + sin_t*sin_part


# ---------------------------------------------------------------- the call

def make_response(wdm, consts, dtype=jnp.float32, filters="polyw", links_per_launch=1):
    r"""
    The jitted GPU response ``f(theta, phi, wp, wc)`` -> (num_links, Nt, Nf)
    in float64, with the shared-sum weights baked in, and its stages for
    timing. ``filters``: ``"polyw"`` (per-row weights from the polynomial
    filters in XLA, then the stored-weight kernel; needs ``consts["poly"]``),
    ``"poly"`` (filters formed inside the kernel) or ``"table"`` (weights
    from the interpolated tables). ``links_per_launch`` > 1 (polyw only)
    builds the weights and runs the rate kernel for that many links at once:
    less per-link glue (all six: 139 against 163 ms on an A400 at Nf=2500),
    more memory (all six do not fit 4 GB at Nf <= 1024).
    """
    amp_w, sinc_w = np.asarray(consts["amp"]), np.asarray(consts["sinc"])
    cast = jax.jit(lambda x: x.astype(dtype))

    @jax.jit
    def common(consts, theta, phi, wp, wc):
        rows = fs._geometry(wdm, consts, theta, phi)
        acc = shared_sums_kernel5(amp_w, sinc_w, jnp.stack([wp, wc]).astype(dtype))
        (zp, sp), (zc, sc) = _recombine_sums5(acc[0]), _recombine_sums5(acc[1])
        return rows, (zp, sp, zc, sc)

    @jax.jit
    def grids(consts, rows, wp, wc, sums):
        rest, amp_row_p, amp_row_c, tau = rows
        grad = lambda q: jnp.gradient(q, wdm.dT, axis=-1)
        f = lambda x: x.astype(dtype)
        return fs.link_grids_minor(
            f(consts["freqs"]), wdm.dF, f(amp_row_p), f(amp_row_c), f(tau),
            f(grad(amp_row_p)), f(grad(amp_row_c)), f(grad(tau)), f(wp), f(wc), *sums)

    nodes = fs.table_nodes(wdm, consts["filters"].shape[-1])
    coef = np.asarray(consts["poly"]) if filters == "poly" else None

    @jax.jit
    def link_poly(rest, grid):
        """One link with the in-kernel polynomial filters."""
        rv = poly_rowvals(rest, wdm.dT, wdm.dF).astype(dtype)
        K = rate_shift_kernel_poly(rv[None], grid[None], wdm.dF, coef,
                                   tile=RATE_TILE, warps=RATE_WARPS)[0]
        return _recombine_shift(K, rest, wdm.dF).astype(jnp.float64)

    @jax.jit
    def link_polyw(consts, rest, grid):
        """One link with per-row weights built from the polynomials in XLA,
        then the stored-weight kernel."""
        r = rest.astype(dtype)
        filters = fs.row_filters_poly(consts["poly"].astype(dtype), r,
                                      jnp.gradient(rest, wdm.dT).astype(dtype))
        w = fs.shift_weights_vec(wdm, filters, r)
        if CARRIER_IN_KERNEL:
            K = rate_shift_kernel(w[None], grid[None], wdm.dF, by_sigma=True,
                                  tile=RATE_TILE, warps=RATE_WARPS,
                                  cycles=(wdm.dF*rest)[None])[0]
            return _recombine_carried(K).astype(jnp.float64)
        K = rate_shift_kernel(w[None], grid[None], wdm.dF, by_sigma=True,
                              tile=RATE_TILE, warps=RATE_WARPS)[0]
        return _recombine_shift(K, rest, wdm.dF).astype(jnp.float64)

    @jax.jit
    def links_batched(consts, rest, grids):
        """All links' weights and rate kernel in one launch (polyw, carried)."""
        r = rest.astype(dtype)
        w = jnp.stack([fs.shift_weights_vec(wdm, fs.row_filters_poly(
            consts["poly"].astype(dtype), r[j],
            jnp.gradient(rest[j], wdm.dT).astype(dtype)), r[j])
            for j in range(rest.shape[0])])
        K = rate_shift_kernel(w, grids, wdm.dF, by_sigma=True, tile=RATE_TILE,
                              warps=RATE_WARPS)
        return jnp.stack([_recombine_shift(K[j], rest[j], wdm.dF)
                          for j in range(rest.shape[0])]).astype(jnp.float64)

    @jax.jit
    def link(consts, rest, grid):
        """One link's weights, kernel and recombination: the working set of
        one link only, which a 4 GB card needs at small Nf."""
        r = rest.astype(dtype)
        filters = fs.row_filters_local(wdm, consts["filters"].astype(dtype),
                                       consts["rate"].astype(dtype), r,
                                       jnp.gradient(rest, wdm.dT).astype(dtype), nodes)
        w = fs.shift_weights(wdm, filters, r)
        K = rate_shift_kernel(w[None], grid[None], wdm.dF, by_sigma=True,
                              tile=RATE_TILE, warps=RATE_WARPS)[0]
        return _recombine_shift(K, rest, wdm.dF).astype(jnp.float64)

    def shifts(rows, grids):
        if filters == "poly":
            return jnp.stack([link_poly(rows[0][j], grids[j])
                              for j in range(grids.shape[0])])
        if filters == "polyw_batched" or (filters == "polyw" and links_per_launch > 1):
            n = links_per_launch if filters == "polyw" else grids.shape[0]
            return jnp.concatenate([links_batched(consts, rows[0][i:i + n], grids[i:i + n])
                                    for i in range(0, grids.shape[0], n)])
        if filters == "polyw":
            return jnp.stack([link_polyw(consts, rows[0][j], grids[j])
                              for j in range(grids.shape[0])])
        return jnp.stack([link(consts, rows[0][j], grids[j])
                          for j in range(grids.shape[0])])

    def response(theta, phi, wp, wc):
        wp, wc = cast(wp), cast(wc)
        rows, sums = common(consts, theta, phi, wp, wc)
        return shifts(rows, grids(consts, rows, wp, wc, sums))

    return response, dict(common=common, grids=grids, shifts=shifts, cast=cast)


def correction_weights(wdm, consts, rest, dtype=jnp.float32):
    """`fast_split.shift_weights_vec` of the delay-rate terms alone (T, T' zeroed)."""
    r = rest.astype(dtype)
    filters = fs.row_filters_poly(consts["poly"].astype(dtype), r,
                                  jnp.gradient(rest, wdm.dT).astype(dtype))
    return fs.shift_weights_vec(wdm, filters.at[:2].set(0.0), r)


def correction_shift(wdm, consts, rest, grid, dtype):
    r"""
    The delay-rate correction of one link's shift in ``dtype`` (float16 for
    the test): grid and weights scaled to unit maxima first, so float16's
    range holds them, the rate kernel, the recombination in float32, then
    the scales put back.
    """
    w = correction_weights(wdm, consts, rest)
    m_max = wdm.Nf*wdm.dF
    sw = jnp.maximum(jnp.abs(w[:, :, :2]).max(), m_max*jnp.abs(w[:, :, 2:]).max())
    sg = jnp.abs(grid).max()
    K = rate_shift_kernel((w/sw).astype(dtype)[None], (grid/sg).astype(dtype)[None],
                          wdm.dF, by_sigma=True, tile=RATE_TILE, warps=RATE_WARPS)[0]
    return _recombine_shift(K.astype(jnp.float32), rest, wdm.dF)*(sw*sg).astype(jnp.float32)
