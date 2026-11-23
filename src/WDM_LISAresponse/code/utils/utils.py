"""
Utility functions for WDM_LISAresponse.

Everything here is support machinery rather than physics: padding and cutting,
mismatch, argument resolution for the response API, generic interpolation on a
uniform grid, and the memory budgeting that decides how the response blocks
its work. The response model itself - the link geometry, the sky basis and the
one-arm kernel - lives in ``lisa_response.response_sub``.

Padding is a property of the grid a record will be analysed on rather than of
the waveform, so it lives here and not in ``waveforms``, and it applies just
as well to data that never came from a generator.

Four functions cover it, and they split cleanly into *decide* and *apply* -
twice, once on the way in and once on the way out. The deciders are pure
arithmetic and touch no data; the appliers do nothing but index::

                     decide                    apply
    going in    wdm_padding_info   ->   apply_zero_padding      (time domain)
    coming out  wdm_cut_info       ->   apply_wdm_cut           (wavelet domain)

The usual sequence is therefore::

    info = wdm_padding_info(len(times), Nf, max_lag_L=L,
                            taper_samples=tukey_taper_samples(len(times), alpha))
    times, h_plus, h_cross = apply_zero_padding(
        times, h_plus, h_cross, info["pad_left"], info["pad_right"])

    # ... build WDM_transform(N=info["target_length"]), run the response ...

    interior = apply_wdm_cut(response, wdm_cut_info(info))

Splitting each stage in two is what makes the pair usable beyond the
``(times, h_plus, h_cross)`` triple: the deciders work on a length and a
dictionary, so a single noise series, one polarisation, or several arrays that
must share one grid are all covered by computing the plan once and applying it
however you like.

The ``info`` dictionary is the only thing crossing between the domains. It
carries the one padding decision in both unit systems - samples for the time
series that was padded, rows for the coefficient grid that will be cut - and,
when the record was windowed, how far each taper ramp reaches. The cut keeps
the rows that hold neither a pad nor a tapered sample.

None of it is applied automatically anywhere. Padding is the caller's
decision, since the two response paths need different amounts of it and the
time-domain path needs no guard band at all.

When a record padded one way must sit on a grid padded another - one
time-domain reference compared against several tilings - ``apply_repadding``
moves it between the two without recomputing it.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def apply_zero_padding(times, h_plus, h_cross, pad_left : int, pad_right : int):
    r"""
    Apply zero padding to two polarisations, extending the times to match.

    **Applies; decides nothing.** It pads by exactly the sample counts it is
    given and has no notion of the wavelet grid, the tiling or the guard band.

    Two ways to reach it. On the *wavelet* path the counts come from
    ``wdm_padding_info``, which works them out from :math:`N_f` and
    :math:`L`::

        info = wdm_padding_info(len(times), Nf, max_lag_L=L)
        times, h_plus, h_cross = apply_zero_padding(
            times, h_plus, h_cross, info["pad_left"], info["pad_right"])

    On the *time-domain* path there is nothing to work out - no wrap to guard
    against and no tiling to satisfy - so pass whatever margin you judge right
    and call this alone.

    Parameters
    ----------
    times : ndarray
        Uniformly sampled times, in seconds. Array shape=(N,).
    h_plus, h_cross : ndarray
        Polarisations on that grid. Array shape=(N,).
    pad_left, pad_right : int
        Zeros to prepend and append, in samples.

    Returns
    -------
    times : ndarray
        Times extended at the same cadence. Array shape=(N+pad_left+pad_right,).
    h_plus, h_cross : ndarray
        Zero padded polarisations, same shape as ``times``.
    """
    times = np.asarray(times)
    h_plus = np.asarray(h_plus)
    h_cross = np.asarray(h_cross)

    if times.ndim != 1 or h_plus.ndim != 1 or h_cross.ndim != 1:
        raise ValueError("times, h_plus, and h_cross must all be 1D arrays.")

    if not (len(times) == len(h_plus) == len(h_cross)):
        raise ValueError("times, h_plus, and h_cross must have the same length.")

    if len(times) < 2:
        raise ValueError("Need at least two time samples to infer dt for padding.")

    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative.")

    dt = times[1] - times[0]
    if not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly sampled to apply zero padding.")

    h_plus_pad = np.pad(h_plus, (pad_left, pad_right), mode="constant", constant_values=0.0)
    h_cross_pad = np.pad(h_cross, (pad_left, pad_right), mode="constant", constant_values=0.0)

    if pad_left == 0 and pad_right == 0:
        return times.copy(), h_plus_pad, h_cross_pad

    times_left = times[0] - dt * np.arange(pad_left, 0, -1)
    times_right = times[-1] + dt * np.arange(1, pad_right + 1)
    times_pad = np.concatenate((times_left, times, times_right))

    return times_pad, h_plus_pad, h_cross_pad


def apply_repadding(series,
                    pad_left_from : int,
                    pad_right_from : int,
                    pad_left_to : int,
                    pad_right_to : int):
    r"""
    Move a padded record onto a grid padded differently, at the same cadence.

    **Applies; decides nothing.** Both paddings are sample counts, as
    ``apply_zero_padding`` takes them. The record moves sample for sample and
    nothing is interpolated, so the two grids must share a cadence and start
    the unpadded signal on the same sample of the absolute timeline.

    This is what lets one time-domain reference serve many wavelet tilings.
    ``wdm_padding_info`` pads by whole rows, so every :math:`N_f` asks for a
    different record, and rerunning the slow reference for each would dominate
    any sweep. The response is local in time - outside the signal plus one
    retarded-phase excursion, about 500 s, it is exactly zero - so compute it
    once on a generously padded record and move it onto each grid here::

        ref = wdm_padding_info(len(times), Nf_ref, extra_rows_left=..., ...)
        # ... time-domain reference on the grid ``ref`` describes ...
        info = wdm_padding_info(len(times), Nf, max_lag_L=L)
        on_grid = apply_repadding(reference, ref["pad_left"], ref["pad_right"],
                                  info["pad_left"], info["pad_right"])

    Growing a pad adds zeros. Shrinking one drops samples, which is allowed
    only when every dropped sample is exactly zero: a crop that discards
    response would silently change the record, so it raises instead.

    Parameters
    ----------
    series : ndarray
        Padded record, shape ``(..., pad_left_from + n + pad_right_from)``.
        Leading axes, such as links or a batch, pass through untouched.
    pad_left_from, pad_right_from : int
        Padding ``series`` carries at each end, in samples.
    pad_left_to, pad_right_to : int
        Padding wanted at each end, in samples.

    Returns
    -------
    repadded : ndarray
        The same record on the new grid. Array shape
        ``(..., pad_left_to + n + pad_right_to)``.
    """
    series = np.asarray(series)
    pads = tuple(int(pad) for pad in
                 (pad_left_from, pad_right_from, pad_left_to, pad_right_to))

    if min(pads) < 0:
        raise ValueError(f"Padding must be non-negative, got {pads}.")

    pad_left_from, pad_right_from, pad_left_to, pad_right_to = pads
    length_from = series.shape[-1]

    if length_from - pad_left_from - pad_right_from < 1:
        raise ValueError(
            f"The padding ({pad_left_from} + {pad_right_from} samples) leaves "
            f"nothing of a record {length_from} samples long."
        )

    # The span of the source that survives; anything outside it is dropped.
    keep_start = max(pad_left_from - pad_left_to, 0)
    keep_stop = length_from - max(pad_right_from - pad_right_to, 0)

    if np.any(series[..., :keep_start]) or np.any(series[..., keep_stop:]):
        raise ValueError(
            "Shrinking the padding would discard non-zero samples. The record "
            "holds response outside the span the new padding keeps; compute "
            "it on a record padded at least as generously as the target."
        )

    widths = [(0, 0)] * (series.ndim - 1) + [
        (max(pad_left_to - pad_left_from, 0), max(pad_right_to - pad_right_from, 0))]

    return np.pad(series[..., keep_start:keep_stop], widths)


def wdm_padding_info(n_samples : int,
                     Nf : int,
                     *,
                     max_lag_L : int = None,
                     extra_rows_left : int = 0,
                     extra_rows_right : int = 0,
                     taper_samples : int = 0):
    r"""
    Work out the zero padding a record needs for the wavelet response.

    **Decides; applies nothing.** It takes a *length*, not an array, and
    returns a dictionary of counts. Hand ``pad_left`` and ``pad_right`` to
    ``apply_zero_padding`` to act on them, and keep the dictionary: it is the
    record ``wdm_cut_info`` reads afterwards to undo exactly this padding.

    Taking a length rather than arrays is what makes it general. A single
    noise series, one polarisation, or several records that must land on the
    same grid are all handled by computing the plan once and applying it
    yourself.

    Three constraints have to hold at once, and they are easiest to satisfy
    together rather than in stages.

    *The tiling.* ``WDM_transform`` needs :math:`N` an even multiple of
    :math:`N_f`, so the total length is rounded up to a multiple of
    :math:`2 N_f`.

    *The wrap.* ``apply_variable_time_shift`` wraps periodically rather than
    running off the record: pixel row :math:`n` reads rows :math:`n-L \ldots
    n+L`, the kernel having hard support at :math:`\pm L` and nothing beyond.
    Padding *both* ends with :math:`\lceil L/2 \rceil` rows is enough to make
    every wrapped read land on zeros - the first physical row reads down to
    :math:`P-L`, which wraps to :math:`N_t+P-L`, and that sits inside the
    trailing pad whenever :math:`2P \geq L`. Padding one end only would cost
    the full :math:`L`.

    *The parity.* The leading pad is rounded up to an even number of rows, so
    that grids padded differently stay comparable coefficient by coefficient.
    Bookkeeping rather than correctness: the basis alternates cosine-like and
    sine-like packets along :math:`n`, so an odd offset measures the signal
    against the other kind - exactly, and reversibly, but as different numbers.
    It costs at most one row and no response calculation is affected.

    Padding is quantised to whole wavelet rows so the signal starts on a row
    boundary, which is what lets ``wdm_cut_info`` recover the original span by
    exact integer slicing rather than by converting samples to rows with a
    ceiling.

    Parameters
    ----------
    n_samples : int
        Length of the unpadded record, in samples.
    Nf : int
        Number of WDM frequency bands. One wavelet row is ``Nf`` samples.
    max_lag_L : int
        Maximum retained lag :math:`L` of the time-delay filter tables. When
        given, :math:`\lceil L/2 \rceil` guard rows are added at each end;
        when ``None`` no guard band is added and only the tiling constraint is
        applied. Optional.
    extra_rows_left, extra_rows_right : int
        Further rows to pad, over and above the guard band. Optional.
    taper_samples : int
        Samples the window taper covers at each end of the record -
        ``waveforms.tukey_taper_samples`` for a Tukey window. It changes no
        padding; it is recorded so ``wdm_cut_info`` drops the tapered rows
        too, and must describe the window actually applied. Optional.

    Returns
    -------
    info : dict
        The padding decision, in both unit systems at once:

        - ``pad_left``, ``pad_right``, ``target_length``, ``input_length``,
          ``taper_samples`` - samples, for the time series that is about to
          be padded;
        - ``rows_left``, ``rows_right``, ``guard_rows``, ``Nt``, ``Nf`` -
          wavelet rows, for the coefficient grid that will later be cut.

        Build the transform with ``N=target_length``, and pass the whole
        dictionary to ``wdm_cut_info`` when the response comes back.
    """
    n_samples_i = int(n_samples)
    Nf_i = int(Nf)
    extra_left_i = int(extra_rows_left)
    extra_right_i = int(extra_rows_right)

    if n_samples_i < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples_i}.")
    if Nf_i < 1:
        raise ValueError(f"Nf must be >= 1, got {Nf_i}.")
    if extra_left_i < 0 or extra_right_i < 0:
        raise ValueError(
            f"extra_rows_left/right must be non-negative, got "
            f"{extra_left_i} and {extra_right_i}."
        )

    taper_i = int(taper_samples)
    if not 0 <= 2 * taper_i <= n_samples_i:
        raise ValueError(
            f"taper_samples must lie in [0, n_samples/2], got {taper_i} "
            f"for {n_samples_i} samples."
        )

    if max_lag_L is None:
        guard_rows = 0
    else:
        max_lag_L_i = int(max_lag_L)
        if max_lag_L_i < 0:
            raise ValueError(f"max_lag_L must be non-negative, got {max_lag_L_i}.")
        guard_rows = -(-max_lag_L_i // 2)          # ceil(L/2), integer only

    rows_left = guard_rows + extra_left_i
    rows_left += rows_left % 2                     # parity: leading pad even
    rows_right = guard_rows + extra_right_i

    pad_left = rows_left * Nf_i

    # The right pad absorbs the tiling remainder on top of its own rows, so
    # the signal keeps the exact row offset ``rows_left`` names.
    provisional = pad_left + n_samples_i + rows_right * Nf_i
    block = 2 * Nf_i
    target_length = int(-(-provisional // block) * block)
    pad_right = int(target_length - n_samples_i - pad_left)

    return {
        "input_length": n_samples_i,
        "target_length": target_length,
        "pad_left": int(pad_left),
        "pad_right": pad_right,
        "rows_left": int(rows_left),
        "rows_right": int(pad_right // Nf_i),
        "guard_rows": int(guard_rows),
        "Nf": Nf_i,
        "Nt": int(target_length // Nf_i),
        "taper_samples": taper_i,
    }


def wdm_cut_info(pad_info : dict,
                 *,
                 extra_rows : int = 0,
                 drop_m0 : bool = True):
    r"""
    Work out which rows and bands to keep in a ``(..., Nt, Nf)`` grid.

    **Decides; applies nothing.** It reads the record ``wdm_padding_info``
    left behind and returns ``slice`` objects. Pass them to
    ``apply_wdm_cut``, or use them directly - ``wavelet_times[time_slice]``
    trims the time axis the same way, which is why the slices are returned
    rather than applied here.

    Drops the padding rows and, when ``wdm_padding_info`` recorded
    ``taper_samples``, every row holding a tapered sample.

    Parameters
    ----------
    pad_info : dict
        The dictionary ``wdm_padding_info`` returned for this record.
    extra_rows : int
        Additional rows to drop at each end, beyond the padding and the
        taper. Optional.
    drop_m0 : bool
        Drop the :math:`m=0` band, which the WDM time shift does not cover.
        Optional.

    Returns
    -------
    info : dict
        ``time_slice`` and ``freq_slice`` for indexing the grid, alongside the
        row counts they were built from.
    """
    extra_i = int(extra_rows)
    if extra_i < 0:
        raise ValueError(f"extra_rows must be non-negative, got {extra_i}.")

    Nt = int(pad_info["Nt"])
    Nf = int(pad_info["Nf"])
    taper = int(pad_info["taper_samples"])

    # The signal's span in samples, [first, end). It starts on a row
    # boundary; it need not end on one.
    first = int(pad_info["pad_left"])
    end = first + int(pad_info["input_length"])

    # Keep a row only if it holds no padding-only and no tapered sample.
    # Without a taper this is the padding cut alone, and a last row holding
    # the signal's final few samples stays.
    start = -(-(first + taper) // Nf) #ceil function
    stop = (end - taper) // Nf if taper else Nt - int(pad_info["rows_right"])

    start += extra_i
    stop -= extra_i

    if stop <= start:
        raise ValueError(
            f"Nothing left after cutting: {start=}, {stop=}, {Nt=}. The "
            f"padding, taper ({taper} samples) and extra_rows together cover "
            f"the whole record."
        )
    
    if drop_m0:
        freq_slice = slice(1, None)
    else:
        freq_slice = slice(None)

    return {
        "wdm_Nt": Nt,
        "Nf": Nf,
        "rows_cut_left": start,
        "rows_cut_right": Nt - stop,
        "time_slice": slice(start, stop),
        "freq_slice": freq_slice,
    }


def apply_wdm_cut(coeffs, cut_info : dict):
    r"""
    Apply the slices from ``wdm_cut_info`` to a coefficient grid.

    **Applies; decides nothing.** It is ``coeffs[..., time_slice,
    freq_slice]``, and exists so the caller does not have to unpack the
    dictionary by hand.

    The grid is indexed as it arrives, so a device array stays on the device
    and a ``numpy`` one stays on the host. Leading batch and link axes pass
    through untouched; the cut always applies to the trailing ``(Nt, Nf)``
    pair.

    Parameters
    ----------
    coeffs : jnp.ndarray or np.ndarray
        Coefficient grid, shape ``(..., Nt, Nf)``.
    cut_info : dict
        The dictionary returned by ``wdm_cut_info``.

    Returns
    -------
    cut : jnp.ndarray or np.ndarray
        The interior of ``coeffs``, same type as the input.
    """
    if coeffs.ndim < 2:
        raise ValueError(
            f"coeffs must have at least two axes, got shape {coeffs.shape}."
        )

    return coeffs[..., cut_info["time_slice"], cut_info["freq_slice"]]


def wdm_mismatch(w1, w2, *, weights=None):
    r"""
    Compare two WDM coefficient grids.

    Returns the sign-maximised mismatch together with the direct residual, so
    one call covers both questions usually asked of two grids: how similar are
    they in shape, and how far apart are they outright.

    .. math::

        \mathrm{mismatch} = 1 - \frac{\left| \langle w_1, w_2 \rangle \right|}
                                     {\sqrt{\langle w_1,w_1 \rangle
                                            \langle w_2,w_2 \rangle}}

    evaluated through the mathematically equivalent normalised residual

    .. math::

        \mathrm{mismatch} = \tfrac{1}{2}
            \left\| \hat{w}_1 - s\,\hat{w}_2 \right\|^2 , \qquad
        s = \mathrm{sign}\langle \hat{w}_1, \hat{w}_2 \rangle .

    The two agree in exact arithmetic, but the second computes the small
    number directly instead of subtracting two numbers that are both nearly
    one. That matters here: measured against a known answer the direct form
    is 2% low at a mismatch of :math:`5\times10^{-15}` and returns exactly
    zero by :math:`5\times10^{-17}`, while this form tracks the truth
    throughout. Comparing two implementations that agree deeply is exactly
    when you need it.

    *Sign*, not phase. WDM coefficients are real, so the only global freedom
    is :math:`\pm 1`, and that is what is maximised over. A constant phase
    shift of the underlying signal is **not** a constant rescaling of its
    wavelet coefficients - for a narrowband signal it is approximately a time
    shift, which is ``WDM_transform.apply_variable_time_shift``, not a
    multiply. So there is no phase to maximise over at this level, and a grid
    that differs from another by a genuine phase offset will register as a
    real mismatch rather than being aligned away.

    Parameters
    ----------
    w1, w2 : ndarray
        WDM coefficient grids of identical shape. Any shape is accepted; the
        inner product runs over every element.
    weights : ndarray
        Per-pixel weights for the inner product, broadcastable to the shape of
        the grids - :math:`1/S_n` to make this a noise-weighted mismatch, one
        value per frequency band being the usual case. ``None`` gives the
        unweighted inner product, which is what you want when validating one
        implementation against another rather than asking a detection
        question. Optional.

    Returns
    -------
    result : dict
        ``mismatch``
            :math:`1 - |\langle \hat{w}_1, \hat{w}_2 \rangle|`, in
            :math:`[0, 1]`, sign-maximised.
        ``overlap``
            ``1 - mismatch``, carried for convenience.
        ``correlation``
            The *signed* normalised inner product, before the sign is taken.
            A negative value means the two grids are anticorrelated, which the
            mismatch alone would hide.
        ``rel_L2``
            :math:`\|w_1 - w_2\| / \|w_1\|`, the direct residual with no
            normalisation or sign alignment. Sensitive to overall scale, which
            the mismatch deliberately is not, so a small mismatch beside a
            large ``rel_L2`` means the shapes agree and the amplitudes do not.
        ``norm1``, ``norm2``
            :math:`\langle w_i, w_i \rangle`, the unnormalised squared norms.

        Every entry is ``nan`` when either grid is non-finite or has zero
        norm, rather than raising.
    """
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)

    if w1.shape != w2.shape:
        raise ValueError(f"Shape mismatch: {w1.shape} vs {w2.shape}.")

    if weights is None:
        def inner(a, b):
            return float(np.dot(a.reshape(-1), b.reshape(-1)))
    else:
        weights = np.asarray(weights, dtype=float)

        try:
            weights = np.broadcast_to(weights, w1.shape)
        except ValueError:
            raise ValueError(
                f"weights of shape {weights.shape} cannot broadcast to the "
                f"grids' shape {w1.shape}."
            )

        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite and non-negative.")

        def inner(a, b):
            return float(np.sum(weights * a * b))

    nan_result = {
        "mismatch": np.nan,
        "overlap": np.nan,
        "correlation": np.nan,
        "rel_L2": np.nan,
        "norm1": np.nan,
        "norm2": np.nan,
    }

    if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))):
        return nan_result

    norm1 = inner(w1, w1)
    norm2 = inner(w2, w2)

    if not np.isfinite(norm1) or not np.isfinite(norm2) \
            or norm1 <= 0.0 or norm2 <= 0.0:
        return nan_result

    # Normalise first, so the residual below is between unit vectors and its
    # squared norm is the mismatch directly.
    hat1 = w1 / np.sqrt(norm1)
    hat2 = w2 / np.sqrt(norm2)

    correlation = inner(hat1, hat2)
    if not np.isfinite(correlation):
        return nan_result

    # The only global freedom real coefficients have.
    sign = -1.0 if correlation < 0.0 else 1.0

    residual = hat1 - sign * hat2
    mismatch = 0.5 * inner(residual, residual)

    if not np.isfinite(mismatch):
        return nan_result

    # A norm cannot be negative; only guard the last bit of rounding.
    if mismatch < 0.0:
        if mismatch > -1e-15:
            mismatch = 0.0
        else:
            return nan_result

    difference = w1 - w2

    return {
        "mismatch": float(mismatch),
        "overlap": float(1.0 - mismatch),
        "correlation": float(correlation),
        "rel_L2": float(np.sqrt(inner(difference, difference) / norm1)),
        "norm1": float(norm1),
        "norm2": float(norm2),
    }


class Link(NamedTuple):
    r"""
    One directed link, from ``sender`` to ``receiver``.

    A labelled pair, and nothing more - it holds *one* link, never several.
    Its whole job is to be distinguishable from a sequence of indices, because
    a bare ``(2, 1)`` cannot be: it is equally readable as the link
    :math:`2 \to 1` and as "the links at positions 2 and 1". Naming the type
    settles it, and settles it in a way that survives a config file, since a
    tuple written to JSON comes back as a list.

    Being a ``NamedTuple`` it still unpacks as ``sender, receiver = link`` and
    compares equal to the plain tuple, so a ``link_convention`` built from
    either works.

    Parameters
    ----------
    sender : int
        Emitting spacecraft, 1-based.
    receiver : int
        Receiving spacecraft, 1-based.
    """
    sender : int
    receiver : int

    def __repr__(self) -> str:
        return f"Link({self.sender}->{self.receiver})"


def resolve_link_index(specifier, link_convention : tuple) -> int:
    r"""
    Resolve one link specifier to its index in a link convention.

    Parameters
    ----------
    specifier : int or Link
        Either a position in ``link_convention``, negative counting from the
        end as in ``numpy``, or a ``Link`` naming the link directly.
    link_convention : tuple
        Directed links, in the order that indexes axis 0 of the geometry
        arrays.

    Returns
    -------
    index : int
        Position of the link in ``link_convention``.
    """
    num_links = len(link_convention)

    # Link before int: a NamedTuple is not an int, but check it first anyway
    # so the intent of the ordering is visible.
    if isinstance(specifier, Link):
        assert specifier in link_convention, \
                    f"{specifier!r} is not in the link convention " \
                    f"{link_convention}. Build a LISAresponse whose " \
                    f"link_convention contains it, or pass an index instead."

        return link_convention.index(specifier)

    if isinstance(specifier, (int, np.integer)):
        index = int(specifier)

        assert -num_links <= index < num_links, \
                    f"Link index must lie in [-{num_links}, {num_links}), " \
                    f"got {index}."

        return index % num_links

    raise TypeError(
        f"A link specifier is an int index or a Link, got {specifier!r}. "
        f"Write Link(sender, receiver) to name a link directly."
    )


def resolve_link_selection(link, link_convention : tuple):
    r"""
    Resolve the ``link`` argument of the response methods.

    Two independent rules, one for *which* links and one for the *shape* they
    come back in:

    1. A ``Link`` names a link directly; an ``int`` names it by position.
       Anything else is a sequence of those.
    2. A single specifier drops the link axis from the result; a sequence
       keeps it - the same way ``array[0]`` drops an axis where ``array[[0]]``
       keeps it.

    So ``Link(2, 1)`` is one link and the result has no link axis, ``[2, 1]``
    is two links by position, and ``[Link(2, 1)]`` is one link *with* the axis.

    Note that a bare tuple is a sequence like any other: ``(2, 1)`` means the
    links at positions 2 and 1, exactly as ``[2, 1]`` does. This is why
    ``Link`` exists - a plain pair cannot say which of the two it means, and
    guessing from whether the caller wrote a tuple or a list breaks the moment
    the value passes through JSON. The one input that would be genuinely
    ambiguous, a bare two-integer tuple that is also a link in the convention,
    is rejected rather than guessed at.

    This lives here rather than on ``LISAresponse`` because it is pure lookup
    that happens to need one attribute. As a method it could only be exercised
    by building a whole instance, which means an orbit provider and a WDM
    transform; as a function it is testable on a bare tuple.

    Parameters
    ----------
    link : None or int or Link or sequence
        ``None`` selects every link in ``link_convention``. Otherwise an
        index, a ``Link``, or a sequence of either.
    link_convention : tuple
        Directed links, as ``Link`` or as plain pairs.

    Returns
    -------
    indices : tuple
        Positions in ``link_convention``, in the requested order.
    drop_axis : bool
        Whether the caller should drop the link axis from its output.
    """
    if link is None:
        return tuple(range(len(link_convention))), False

    if isinstance(link, (Link, int, np.integer)):
        return (resolve_link_index(link, link_convention),), True

    # The only ambiguous input there is. Refuse it rather than pick a reading:
    # under the rules above it means two indices, but it is exactly what
    # someone reaching for a single link would write.
    if (isinstance(link, tuple) and len(link) == 2
            and all(isinstance(entry, (int, np.integer)) for entry in link)
            and tuple(link) in tuple(tuple(entry) for entry in link_convention)):
        raise ValueError(
            f"{link} is ambiguous: it is both a link in the convention and a "
            f"pair of indices. Write Link{link} for the link "
            f"sender {link[0]} -> receiver {link[1]}, or [{link[0]}, "
            f"{link[1]}] for the links at those two positions."
        )

    indices = tuple(
        resolve_link_index(entry, link_convention) for entry in link
    )

    assert len(indices) > 0, \
                "At least one link must be selected, got an empty link."

    return indices, False


    return indices, False


#: Widest Lagrange stencil ``interpolate_uniform_grid`` will accept. The
#: outermost weight divides by :math:`(n-1)!`, held as a Python int, and
#: :math:`21! = 5\times10^{19}` exceeds int64 - so twenty-two taps fails
#: inside jax rather than here. Twenty is the last even width that fits.
MAX_INTERP_TAPS = 20


def interpolate_uniform_grid(samples,
                             grid_start : float,
                             grid_step : float,
                             query_times,
                             num_taps : int = 12,
                             extrapolate : bool = False):
    r"""
    Interpolate uniformly sampled series at arbitrary times.

    The samples sit at :math:`t_k = t_{\mathrm{start}} + k\,\delta t`, so the
    sample coordinate of a query time is :math:`(t - t_{\mathrm{start}})/
    \delta t` and no search is needed - the stencil follows from
    :math:`\lfloor \cdot \rfloor` alone.

    The scheme is Lagrange interpolation on ``num_taps`` samples centred on the
    query, with the classical weights

    .. math::

        h(t) = \sum_j w_j(x)\, p_j , \qquad
        w_j(x) = \prod_{m \neq j} \frac{x - o_m}{o_j - o_m} ,

    where :math:`x` is the fractional part of the sample coordinate and
    :math:`o_j` runs over the integer offsets :math:`-n/2+1 \ldots n/2`. The
    denominators are products of integers, known at trace time, so each weight
    costs one running product and a single division by a constant.

    The stencil is local, gathering taps rather than solving a system over
    the whole record, which is what lets the response quadrature evaluate it at
    arbitrary retarded phases.

    Twelve taps is the smallest even count whose error stays below the
    :math:`10^{-4}` local-delay term of the wavelet path across the whole
    usable band, EMRI harmonics included: against an exact sinusoid it gives
    1.4e-5 at 6.7 samples per cycle and ~1e-14 at 33, where the four-tap
    Catmull-Rom this replaced gives 1.5e-2 and 8.0e-5. Sixteen costs 1.6 times
    as much and buys nothing real, saturating against float64 and, sooner,
    against the error the waveform samples already carry.

    By default, taps falling off either end are clamped to the end sample and
    queries outside the sampled span return zero. The two are deliberately
    separate, since dropping out-of-range taps would leave the weights
    unnormalised and ramp a delayed waveform to zero across a cell instead of
    stopping it. That is the right edge for a waveform, which stops at the end
    of its record.

    With ``extrapolate``, the stencil is moved inward near either end so all
    its taps sit on samples, and queries outside the span are extrapolated by
    the same polynomial - the right edge for a smooth quantity that continues
    past the record, such as the detector geometry.

    Parameters
    ----------
    samples : jnp.ndarray
        Sampled series. Array shape=(batch, num_samples).
    grid_start : float
        Time of sample zero, :math:`t_{\mathrm{start}}`, in seconds.
    grid_step : float
        Sample spacing, :math:`\delta t`, in seconds.
    query_times : jnp.ndarray
        Times at which to evaluate, in seconds. Array shape=(batch, num_query),
        sharing the leading axis with ``samples``.
    num_taps : int
        Stencil width. Must be even, at least two, and at most
        ``MAX_INTERP_TAPS``; it is a compile-time constant since it sets the
        loop structure. Optional.
    extrapolate : bool
        Keep the stencil on the samples at the ends and extrapolate outside
        the span, rather than clamping taps and returning zero there. Must be
        a python bool; it sets the code path. Optional.

    Returns
    -------
    values : jnp.ndarray
        Interpolated values. Array shape=(batch, num_query).
    """
    assert num_taps >= 2 and num_taps % 2 == 0, \
                f"num_taps must be even and at least two, got {num_taps=}."

    # The outermost node's weight divides by (num_taps - 1)!, accumulated as a
    # Python int. At twenty-two taps that is 21! = 5e19, past int64, and jax
    # rejects it while tracing with an OverflowError that names neither this
    # function nor the tap count. Refuse it here instead. Raising rather than
    # promoting to float is deliberate: nothing wants a stencil this wide.
    # Lagrange interpolation is ill-conditioned at high order, and twelve taps
    # already saturate against float64 by 33 samples per cycle - sixteen costs
    # 1.6 times as much and buys nothing measurable.
    assert num_taps <= MAX_INTERP_TAPS, \
                f"num_taps must be at most {MAX_INTERP_TAPS}, got " \
                f"{num_taps=}: the Lagrange denominator overflows int64 " \
                f"beyond that. Twelve is the documented default and wider " \
                f"stencils are not better conditioned."

    num_samples = samples.shape[-1]
    half = num_taps // 2
    offsets = tuple(range(-half + 1, half + 1))

    coordinate = (query_times - grid_start) / grid_step
    base = jnp.floor(coordinate)
    if extrapolate:
        assert num_samples >= num_taps, \
                f"extrapolate needs at least {num_taps=} samples, got " \
                f"{num_samples=}."
        # Slide the stencil inward so taps base-half+1 .. base+half all exist;
        # the fraction then leaves [0, 1) and the same weights evaluate the
        # polynomial off-centre, or beyond the span.
        base = jnp.clip(base, half - 1, num_samples - 1 - half)
    fraction = coordinate - base
    index = base.astype(
        jnp.int64 if jax.config.read("jax_enable_x64") else jnp.int32
    )

    values = jnp.zeros_like(fraction)
    for node in offsets:
        # Numerator varies with the query; the denominator is an integer
        # product fixed by the stencil, so it folds away at trace time.
        numerator = jnp.ones_like(fraction)
        denominator = 1
        for other in offsets:
            if other != node:
                numerator = numerator * (fraction - other)
                denominator *= node - other

        taps = jnp.take_along_axis(
            samples, jnp.clip(index + node, 0, num_samples - 1), axis=-1,
        )
        values = values + (numerator / denominator) * taps

    if extrapolate:
        return values
    return jnp.where(
        (coordinate >= 0.0) & (coordinate <= num_samples - 1), values, 0.0,
    )


def resolve_time_chunk_size(num_samples : int,
                            batch_size : int,
                            num_links : int,
                            memory_budget_bytes : float = 2 ** 30) -> int:
    r"""
    Choose a time-block length that keeps working memory inside a budget.

    The time-domain response allocates far more than it returns, and the excess
    is invisible from the API: per block of ``chunk`` reception times it holds
    ``80 * num_links * chunk`` bytes of host geometry - a light-travel time, a
    unit vector and two positions per link and sample - and
    ``8 * batch * num_links * chunk`` per live array of quadrature working set.
    Over four years at 5 s cadence for six links that is 11 GiB of geometry and
    14 GiB of working set against a 1 GiB result, which is why the geometry is
    rebuilt per block rather than once up front.

    Only that working memory is budgeted here. The output is the caller's own
    request, sized ``8 * batch * num_links * num_samples`` and obvious from the
    arguments, so it is left to them exactly as elsewhere in the package rather
    than second-guessed. Neither ``num_integral_points`` nor ``num_taps``
    enters: both are swept one at a time rather than materialised, so they cost
    time and not memory.

    Parameters
    ----------
    num_samples : int
        Length of the reception-time grid, :math:`N`.
    batch_size : int
        Number of sources evaluated together.
    num_links : int
        Number of directed links requested.
    memory_budget_bytes : float
        Ceiling on working memory, in bytes. Optional.

    Returns
    -------
    chunk_size : int
        Block length in samples, at least one and at most ``num_samples``.
    """
    # Live (batch, links, chunk) float64 arrays at the peak of one quadrature
    # node: query times, sample index, fractional part, then per tap a running
    # numerator, a gathered tap and the accumulator, plus slack for the two
    # quadrature sums. Scales with neither the tap count nor the node count,
    # since both are accumulated one at a time. XLA fuses much of it, so this
    # is an upper bound and the sizing it drives is conservative.
    quadrature_working_arrays = 12
    bytes_per_sample = (
        80 * num_links
        + 8 * quadrature_working_arrays * batch_size * num_links
    )

    chunk_size = int(memory_budget_bytes // bytes_per_sample)

    return max(1, min(chunk_size, int(num_samples)))
