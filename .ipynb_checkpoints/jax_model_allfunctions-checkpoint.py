# =============================================================================
# SUPPORT PARAMETERIZATION
# =============================================================================
#
# Support defined as the (soft) intersection of N half-spaces:
#
#     sigma_i(x) = sigmoid((d_i - n_i . x) / eps)
#     S(x)       = prod_i sigma_i(x) = exp(sum_i log sigma_i(x))
#
# All shapes below are for a SINGLE reconstruction instance (no leading batch
# axis). A population of independent restarts is handled by vmapping over
# this module's functions from optimize.py -- vmap composes cleanly with the
# custom_vjp defined here because it is built from ordinary batchable JAX
# primitives (einsum, scan).
#
# Why a custom VJP:
# ------------------
# The naive implementation (`dot = einsum('dhwc,nc->dhwn', coords, n)`)
# materializes a (D, H, W, N) tensor for both `dot` and `sigma`. For the
# largest problem sizes this repo targets (grid ~ 125 x 225 x 225, N up to
# 256) that is:
#
#     6.33M voxels * 256 planes * 4 bytes ~= 6.0 GiB   (per tensor!)
#
# which does not fit comfortably in a 32 GB GPU once FFT buffers and L-BFGS
# history are also live -- and autodiff would keep at least one such tensor
# around for the backward pass. Standard rematerialization (jax.checkpoint)
# does not fix this on its own unless applied at exactly the per-plane
# granularity used below, which is effectively what this custom VJP does by
# hand: we `lax.scan` over the N half-spaces, keeping only a running
# (D, H, W) log-support accumulator, and recompute sigma_i per-plane in the
# backward pass instead of storing it. Peak extra memory becomes O(D*H*W),
# independent of N.
#
# Derivation (per plane i):
#     log S = sum_i log(clip(sigma_i, 1e-6, 1.0))
#     d(log S)/d d_i =  (1 - sigma_i) / eps          (masked to 0 where clipped)
#     d(log S)/d n_i = -(1 - sigma_i) / eps * x       (masked to 0 where clipped)
#     d(log S)/d x   = -(1 - sigma_i) / eps * n_i     (masked to 0 where clipped)
#     dS/d(.)        = S * d(log S)/d(.)
#
# so, given the upstream cotangent g (shape (D,H,W)) and S = exp(log S):
#     w      = (g * S) * active_i * (1 - sigma_i) / eps   # (D, H, W)
#     dS/dd_i =  sum_x w
#     dS/dn_i = -sum_x w * x
#     dS/dx   = -sum_i w * n_i                              # (D, H, W, 3)
#
# `coords` DOES need a real gradient (not `None`): support.py's own usage
# never needs it (coords is a fixed, non-trainable grid there), but
# multi_support.py passes `coords - centers` for a trainable `centers`, and
# the chain rule through that subtraction requires d(support)/d(coords).
# =============================================================================
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def make_coords(grid_shape):
    """Fixed (not trainable) coordinate grid, centered at the origin.

    Returns an array of shape (D, H, W, 3) with the same axis convention as
    the original TensorFlow implementation (x, y, z stacked last, meshgrid
    built from z, y, x with 'ij' indexing).
    """
    D, H, W = grid_shape
    z = jnp.linspace(-(D - 1) / 2.0, (D - 1) / 2.0, D)
    y = jnp.linspace(-(H - 1) / 2.0, (H - 1) / 2.0, H)
    x = jnp.linspace(-(W - 1) / 2.0, (W - 1) / 2.0, W)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    coords = jnp.stack([xx, yy, zz], axis=-1)  # (D, H, W, 3)
    return coords


def _sigma_i(n_i, d_i, coords, eps):
    dot = jnp.einsum("dhwc,c->dhw", coords, n_i)
    return jax.nn.sigmoid((d_i - dot) / eps)


@jax.custom_vjp
def halfspace_support(n, d, coords, eps):
    """Soft polytope indicator, memory O(D*H*W) instead of O(D*H*W*N).

    Parameters
    ----------
    n : (N, 3) array, unit-norm half-space normals.
    d : (N,) array, half-space offsets.
    coords : (D, H, W, 3) array, fixed coordinate grid (see `make_coords`).
    eps : scalar, softness of the half-space sigmoid.

    Returns
    -------
    S : (D, H, W) array in [0, 1].
    """
    logS0 = jnp.zeros(coords.shape[:3], dtype=coords.dtype)

    def step(logS, nd_i):
        n_i, d_i = nd_i
        sigma = _sigma_i(n_i, d_i, coords, eps)
        logS = logS + jnp.log(jnp.clip(sigma, 1e-6, 1.0))
        return logS, None

    logS, _ = lax.scan(step, logS0, (n, d))
    return jnp.exp(logS)


def _halfspace_support_fwd(n, d, coords, eps):
    S = halfspace_support(n, d, coords, eps)
    return S, (n, d, coords, eps, S)


def _halfspace_support_bwd(res, g):
    n, d, coords, eps, S = res
    gS = g * S  # (D, H, W), computed once and reused for every plane

    def step(dcoords_acc, nd_i):
        n_i, d_i = nd_i
        sigma = _sigma_i(n_i, d_i, coords, eps)
        # ** Bug fixed here, found by a targeted finite-difference test: **
        # the forward pass computes log(clip(sigma, 1e-6, 1.0)), so wherever
        # sigma is clipped the TRUE derivative is exactly zero (a clipped
        # value doesn't respond to its input at all) -- but this line
        # previously used the unclipped sigmoid derivative unconditionally,
        # injecting a small but nonzero spurious gradient into every
        # saturated ("deep outside" or "deep inside") voxel for every
        # plane. Verified with sum(S) as an isolating probe (linear in S,
        # so gS = g*S = S exactly): a plane held deep in saturation across
        # an entire 20x18x16 grid gave an analytic gradient of 0.0096
        # (~1.67e-6/voxel) against a finite-difference gradient of exactly
        # 0. That per-voxel error is small, but it scales with total grid
        # VOLUME while genuine boundary-driven gradient signal only scales
        # with grid surface area -- so it is relatively worse at large,
        # realistic BCDI grid sizes than in small unit tests, even though
        # it stayed invisible in this project's original (small-grid)
        # gradient checks.
        active = jnp.logical_and(sigma > 1e-6, sigma < 1.0).astype(sigma.dtype)
        w = gS * active * (1.0 - sigma) / eps    # (D, H, W)
        dd_i = jnp.sum(w)
        dn_i = -jnp.einsum("dhw,dhwc->c", w, coords)
        # ** Second bug, found integrating multi_support.py: ** this
        # function used to unconditionally return `None` for the gradient
        # w.r.t. `coords`, on the assumption that `coords` is always a
        # fixed, non-trainable grid. That's true for the single-part
        # support in this file, but multi_support.py passes
        # `coords - centers` (a per-part, per-voxel shift by a TRAINABLE
        # center), so the `None` was silently discarding the gradient path
        # to `centers` entirely -- confirmed by a finite-difference check
        # that found `centers`' analytic gradient was exactly 0 where it
        # should have been -15.8. d(log sigma_i)/dx = -(1-sigma_i)*n_i/eps
        # (chain rule through z_i = (d_i - n_i.x)/eps), so the per-voxel
        # contribution to d(coords) is `-w * n_i`, accumulated over planes.
        dcoords_acc = dcoords_acc - w[..., None] * n_i   # (D, H, W, 3)
        return dcoords_acc, (dn_i, dd_i)

    dcoords0 = jnp.zeros_like(coords)
    dcoords, (dn, dd) = lax.scan(step, dcoords0, (n, d))
    # eps is treated as a constant (no gradient requested).
    return dn, dd, dcoords, None


halfspace_support.defvjp(_halfspace_support_fwd, _halfspace_support_bwd)


# -----------------------------------------------------------------------
# Sphere-constrained normals: minimal (non-redundant) parameterization.
#
# ** Bug fixed here, found by profiling actual L-BFGS runs: **
# An earlier version of this file used `n = n_raw / ||n_raw||` with
# `n_raw` a free R^3 vector (3 parameters for a 2-DOF constraint). That is
# a valid reparameterization for plain gradient descent -- the gradient of
# f(n_raw/||n_raw||) w.r.t. n_raw is analytically exactly zero along the
# radial direction, verified below -- but it leaves a "gauge" direction
# with literally zero curvature AND zero gradient. Empirically, running
# `optax.lbfgs` on this parameterization let `||n_raw||` drift by ~3.7x
# over 80 steps (e.g. 2.0 -> 7.4) purely from numerical mixing in the
# quasi-Newton history, since nothing in the objective ever pulls it back.
# Because the gradient w.r.t. n_raw scales as 1/||n_raw|| for a fixed
# physical (tangential) effect, this drift silently shrinks the *useful*
# gradient magnitude as optimization proceeds -- an unintended, spurious
# decay of effective step size that Adam's per-coordinate RMS
# normalization mostly shrugs off, but that degrades a quasi-Newton
# method's curvature estimate along that direction (near-singular
# y_k^T s_k), which is likely a major reason L-BFGS underperformed Adam in
# practice on this parameterization.
#
# Fix: parameterize with exactly 2 free parameters per normal (matching
# the sphere's true dimension), via inverse stereographic projection
# through the north pole (0,0,1); p=(0,0) maps to the south pole (0,0,-1).
# This has no flat/gauge direction anywhere except the single excluded
# point at infinity, so there is nothing for a quasi-Newton history to
# "leak" into.
# -----------------------------------------------------------------------
def stereographic_to_unit(p):
    """Inverse stereographic projection: R^2 -> S^2 minus the north pole.

    p : (..., 2) -> n : (..., 3), unit norm.
    """
    p2 = jnp.sum(p ** 2, axis=-1, keepdims=True)
    x = 2.0 * p[..., 0:1]
    y = 2.0 * p[..., 1:2]
    z = p2 - 1.0
    return jnp.concatenate([x, y, z], axis=-1) / (1.0 + p2)


def unit_to_stereographic(n, eps=1e-6):
    """Forward stereographic projection: S^2 minus the north pole -> R^2.

    Only used for initialization (mapping a randomly sampled unit vector
    to its (p1, p2) coordinates); not needed in the training loop.
    """
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    denom = jnp.clip(1.0 - z, eps, None)
    return jnp.stack([x / denom, y / denom], axis=-1)


def init_support_params(key, N, grid_shape, size_factor=4.0):
    """Random init for a single instance's half-space support parameters."""
    key_n, key_d = jax.random.split(key)
    n0 = jax.random.normal(key_n, (N, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    p0 = unit_to_stereographic(n0)
    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    d0 = jnp.ones((N,)) * R
    return {"p_raw": p0, "d": d0}


def compute_support(params, coords, eps):
    """Map the minimal 2-DOF parameterization to a unit normal and evaluate S."""
    n = stereographic_to_unit(params["p_raw"])
    return halfspace_support(n, params["d"], coords, eps)
    
# =============================================================================
# MULTI-PART (NON-CONVEX) SUPPORT
# =============================================================================
# Non-convex support as a soft UNION of M convex parts, each a soft
# INTERSECTION of N_per_part half-spaces -- ported from a draft TensorFlow
# `MultiHalfSpaceMaskSupport`, with the same fixes already applied to the
# single-convex-part support in `support.py`:
#
#   - Minimal (2-DOF) stereographic normal parameterization instead of
#     spherical angles (theta, phi). Spherical angles have a coordinate
#     singularity at the poles (dn/dtheta -> 0 as theta -> 0 or pi) and a
#     latitude-dependent "speed" (the Jacobian scales with sin(theta)),
#     which reintroduces exactly the optimizer-conditioning problem fixed
#     for the single-part case in support.py -- just via a different
#     mechanism (coordinate-chart distortion instead of a redundant flat
#     direction). `theta, phi` in the original are replaced by `p_raw`
#     (M, N, 2) here, mapped through the same `stereographic_to_unit` used
#     for the single-part support.
#   - The per-part intersection reuses `halfspace_support` ITSELF (not a
#     new bespoke implementation) via `jax.vmap` over the M parts. This
#     means every part automatically gets: the scan-based O(D,H,W) memory
#     footprint (independent of N_per_part), and the corrected
#     clip-boundary gradient mask (see the comment in
#     `_halfspace_support_bwd`). Re-deriving a separate (D,H,W,M,N)
#     custom VJP for this file would have risked reintroducing both bugs.
#
# The union of M parts, `1 - prod_m(1 - S_m)`, is left to ordinary
# autodiff: M is small (a handful of domains) -- unlike N, which can be in
# the hundreds -- so materializing a (D,H,W,M) tensor for the union step is
# not the kind of memory concern the (D,H,W,N) tensor was for planes.
import jax
import jax.numpy as jnp

def init_multi_support_params(key, M_parts, N_per_part, grid_shape, size_factor=4.0,
                               use_gates=False):
    """Random init for a single instance's multi-part support parameters."""
    key_n, key_d, key_c, key_g = jax.random.split(key, 4)

    n0 = jax.random.normal(key_n, (M_parts, N_per_part, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    p0 = unit_to_stereographic(n0)

    R = jnp.min(jnp.asarray(grid_shape, dtype=jnp.float32)) / size_factor
    d0 = jnp.ones((M_parts, N_per_part)) * R

    centers0 = jax.random.uniform(key_c, (M_parts, 3), minval=-R, maxval=R)

    params = {"p_raw": p0, "d": d0, "centers": centers0}
    if use_gates:
        # sigmoid(1.0) ~= 0.73: mostly-open gates at init, matching the
        # original's `tf.ones(...)` init for the pre-sigmoid gate logit.
        params["alpha_raw"] = jnp.ones((M_parts,))
    return params


def compute_multi_support(params, coords, eps, use_gates=False):
    """Soft non-convex support: union of M soft convex (half-space
    intersection) parts, each optionally centered independently.

    Parameters
    ----------
    params : dict with "p_raw" (M,N,2), "d" (M,N), "centers" (M,3), and
        optionally "alpha_raw" (M,) if use_gates=True.
    coords : (D, H, W, 3), shared fixed coordinate grid.
    eps : half-space softness.

    Returns
    -------
    support : (D, H, W) array in [0, 1].
    """
    n = stereographic_to_unit(params["p_raw"])          # (M, N, 3)
    d = params["d"]                                       # (M, N)
    centers = params["centers"]                            # (M, 3)

    # (M, D, H, W, 3): each part sees the grid shifted by its own center.
    coords_shifted = coords[None, ...] - centers[:, None, None, None, :]

    def per_part(n_m, d_m, coords_m):
        return halfspace_support(n_m, d_m, coords_m, eps)   # (D, H, W)

    S_parts = jax.vmap(per_part, in_axes=(0, 0, 0))(n, d, coords_shifted)  # (M,D,H,W)

    if use_gates:
        gate = jax.nn.sigmoid(params["alpha_raw"])[:, None, None, None]  # (M,1,1,1)
        S_parts = S_parts * gate

    # Soft union (probabilistic OR): 1 - prod_m(1 - S_m).
    support = 1.0 - jnp.prod(1.0 - S_parts, axis=0)
    return support

# =============================================================================
# PHASE PARAMETERIZATION
# =============================================================================
# Single-instance (no leading batch axis); see support.py for why.


import jax
import jax.numpy as jnp

_TWO_PI = 2.0 * jnp.pi


def _reciprocal_vector(hkl, lattice_matrix):
    hkl = jnp.asarray(hkl, dtype=jnp.float32)
    if lattice_matrix is not None:
        A = jnp.asarray(lattice_matrix, dtype=jnp.float32)
        B = _TWO_PI * jnp.transpose(jnp.linalg.inv(A))
        Q = B @ hkl
    else:
        Q = _TWO_PI * hkl
    return jnp.linalg.norm(Q)


def init_phase_params(key, grid_shape, phase_type="grid", initial_guess=None,
                       hkl=(1, 1, 1), lattice_matrix=None, init_scale=1e-3):
    """Initialize phase parameters and any static (non-trainable) metadata.

    Returns
    -------
    params : dict of trainable arrays for this phase type.
    static : dict of non-trainable metadata (e.g. Qnorm for 'displacement').
    """
    grid_shape = tuple(grid_shape)

    if phase_type == "grid":
        if initial_guess is not None:
            phase0 = jnp.angle(initial_guess)
        else:
            phase0 = jax.random.uniform(key, grid_shape, minval=0.0, maxval=1.0)
        return {"phase": phase0}, {"phase_type": "grid"}

    if phase_type == "phasor":
        return (
            {"c_raw": jnp.ones(grid_shape), "s_raw": jnp.zeros(grid_shape)},
            {"phase_type": "phasor", "eps": 1e-8},
        )

    if phase_type == "displacement":
        Qnorm = _reciprocal_vector(hkl, lattice_matrix)
        if initial_guess is not None:
            phase0 = jnp.angle(initial_guess)
            u0 = phase0 / (Qnorm + 1e-8)
        else:
            u0 = init_scale * jax.random.normal(key, grid_shape)
        return {"u": u0}, {"phase_type": "displacement", "Qnorm": Qnorm}

    raise ValueError(f"Unknown phase_type: {phase_type!r}")


def compute_phase(params, static):
    """Returns a scalar phase field phi, shape (D, H, W)."""
    phase_type = static["phase_type"]
    if phase_type == "grid":
        return params["phase"]
    if phase_type == "displacement":
        return static["Qnorm"] * params["u"]
    raise ValueError(f"{phase_type!r} has no scalar phase; use compute_phasor.")


def compute_phasor(params, static):
    """Returns a (cos, sin) pair, each shape (D, H, W)."""
    phase_type = static["phase_type"]
    if phase_type == "phasor":
        c, s = params["c_raw"], params["s_raw"]
        denom = jnp.sqrt(c ** 2 + s ** 2 + static["eps"])
        return c / denom, s / denom
    if phase_type in ("grid", "displacement"):
        phi = compute_phase(params, static)
        return jnp.cos(phi), jnp.sin(phi)
    raise ValueError(f"Unknown phase_type: {phase_type!r}")

# =============================================================================
# LOSSES
# =============================================================================
# Single-instance (no leading batch axis). The FFT-based fidelity term is
# left to JAX's built-in autodiff: jnp.fft is linear, so its VJP is just an
# (adjoint) FFT with no extra activation storage -- there is nothing a
# hand-written adjoint would improve here.
import jax
import jax.numpy as jnp


def mae(Iobs, Icalc):
    """Normalized mean absolute error."""
    return jnp.sum(jnp.abs(Iobs - Icalc)) / jnp.sum(Iobs)

def mse(Iobs, Icalc):
    """Normalized mean squared error."""
    return jnp.sum((jnp.sqrt(Iobs) - jnp.sqrt(Icalc))**2) / jnp.sum(jnp.sqrt(Iobs))

def poisson_kl(Iobs, Icalc, eps=1e-12):
    """Poisson KL divergence, averaged per voxel."""
    ratio = Iobs / (Icalc + eps)
    kl = Icalc - Iobs + jnp.where(Iobs > 0, Iobs * jnp.log(ratio), 0.0)
    N = Iobs.size
    return jnp.sum(kl) / N


def _center_pad(obj, target_shape):
    pads = []
    for src, dst in zip(obj.shape, target_shape):
        delta = dst - src
        p0 = delta // 2
        p1 = delta - p0
        pads.append((p0, p1))
    return jnp.pad(obj, pads)


def fourier_loss(support, amplitude, phase, Iobs, metric="mae"):
    """Forward model (support * amplitude * phasor -> padded FFT -> |.|^2)
    plus a data-fidelity metric against Iobs.
    """
    modulus = support * amplitude

    if isinstance(phase, tuple):
        c, s = phase
        obj = jax.lax.complex(modulus * c, modulus * s)
    else:
        obj = jax.lax.complex(modulus * jnp.cos(phase), modulus * jnp.sin(phase))

    obj_p = _center_pad(obj, Iobs.shape)

    Icalc = jnp.abs(
        jnp.fft.ifftshift(jnp.fft.fftn(jnp.fft.fftshift(obj_p)))
    ) ** 2

    Iobs = Iobs.astype(jnp.float32)
    Icalc = Icalc.astype(jnp.float32)

    if metric == "mae":
        return mae(Iobs, Icalc)
    if metric == "mse":
        return mse(Iobs, Icalc)
    if metric == "poisson":
        return poisson_kl(Iobs, Icalc)
    raise ValueError(f"Unknown metric: {metric!r}, choose 'mae', 'mse' or 'poisson'.")


def tv_loss_phase(phase, eps=1e-9):
    """Total variation of the phase field (scalar phase or (c, s) phasor)."""

    def diff(x, axis):
        return jnp.diff(x, axis=axis)

    if isinstance(phase, tuple):
        c, s = phase
        dcx, dcy, dcz = diff(c, 0), diff(c, 1), diff(c, 2)
        dsx, dsy, dsz = diff(s, 0), diff(s, 1), diff(s, 2)

        dcx2 = jnp.square(dcx[:, :-1, :-1])
        dcy2 = jnp.square(dcy[:-1, :, :-1])
        dcz2 = jnp.square(dcz[:-1, :-1, :])

        dsx2 = jnp.square(dsx[:, :-1, :-1])
        dsy2 = jnp.square(dsy[:-1, :, :-1])
        dsz2 = jnp.square(dsz[:-1, :-1, :])

        grad_sq = (dcx2 + dcy2 + dcz2) + (dsx2 + dsy2 + dsz2)
        return jnp.mean(grad_sq + eps)

    phi = phase

    def wrapped_diff(p, axis):
        d = jnp.diff(p, axis=axis)
        return jnp.arctan2(jnp.sin(d), jnp.cos(d))

    dx = wrapped_diff(phi, 0)
    dy = wrapped_diff(phi, 1)
    dz = wrapped_diff(phi, 2)

    dx2 = jnp.square(dx[:, :-1, :-1])
    dy2 = jnp.square(dy[:-1, :, :-1])
    dz2 = jnp.square(dz[:-1, :-1, :])

    grad_sq = dx2 + dy2 + dz2
    return jnp.mean(grad_sq + eps)


def small_support(support):
    """Penalty on the size (mass) of the support."""
    return jnp.sum(support)


def total_loss(support, amplitude, phase, Iobs, alpha=0.8, beta=0.1, metric="mae"):
    """Fourier fidelity + support-size penalty + phase-TV penalty.

    Parameters
    ----------
    alpha : weight for the support-size penalty.
    beta  : weight for the phase TV penalty.
    """
    fourier = fourier_loss(support, amplitude, phase, Iobs, metric=metric)
    small = alpha * small_support(support)
    smooth_phase = beta * tv_loss_phase(phase)
    return fourier + small + smooth_phase

# =============================================================================
# MODEL
# =============================================================================

def make_coords_for(Iobs_shape):
    """Convenience: derive grid_shape from an observed-intensity shape the
    same way the original implementation did (grid = Iobs // 2), and build
    the fixed coordinate grid for it.
    """
    D, H, W = Iobs_shape
    grid_shape = (D // 2, H // 2, W // 2)
    return grid_shape, make_coords(grid_shape)


def init_model(key, grid_shape, N=64, size_factor=4.0, phase_type="grid",
               phase_kwargs=None, support_type="single", support_kwargs=None):
    """Initialize a single instance's trainable parameters.

    Parameters
    ----------
    support_type : "single" (default) | "multi"
        "single": one convex object, `N` half-spaces (as before).
        "multi": a non-convex object built as a soft union of several
            convex parts -- see multi_support.py. `support_kwargs` may
            contain "M_parts" (default 2), "N_per_part" (default N),
            "use_gates" (default False, learnable per-part on/off gates).

    Returns
    -------
    params : dict pytree of trainable arrays.
    model_static : dict of non-trainable metadata: "phase_type" (and, for
        'displacement', "Qnorm"), plus "support_type" (and, for "multi",
        "use_gates"). This dict is threaded through `forward`/`loss_fn`
        exactly where `phase_static` used to be -- existing code that only
        reads `model_static["phase_type"]` etc. is unaffected, since keys
        are only ever added here, never renamed or removed.
    """
    key_support, key_phase = jax.random.split(key)
    support_kwargs = support_kwargs or {}

    if support_type == "single":
        support_params = init_support_params(key_support, N, grid_shape, size_factor)
        support_static = {"support_type": "single"}
    elif support_type == "multi":
        M_parts = support_kwargs.get("M_parts", 2)
        N_per_part = support_kwargs.get("N_per_part", N)
        use_gates = support_kwargs.get("use_gates", False)
        support_params = init_multi_support_params(
            key_support, M_parts, N_per_part, grid_shape, size_factor,
            use_gates=use_gates,
        )
        support_static = {"support_type": "multi", "use_gates": use_gates}
    else:
        raise ValueError(f"Unknown support_type: {support_type!r}, choose 'single' or 'multi'.")

    phase_params, phase_static = init_phase_params(
        key_phase, grid_shape, phase_type=phase_type, **(phase_kwargs or {})
    )
    model_static = dict(phase_static)
    model_static.update(support_static)

    params = {"support": support_params, "phase": phase_params}
    return params, model_static


def init_params_only(key, grid_shape, N=64, size_factor=4.0, phase_type="grid",
                      phase_kwargs=None, support_type="single", support_kwargs=None):
    """Like `init_model`, but drops the (non-batchable, string-containing)
    static metadata -- for use under `jax.vmap` when generating a population
    of restarts. `model_static` is identical for every restart by
    construction (it never depends on the random key), so callers should
    fetch it once via `init_model` outside the vmap.
    """
    params, _model_static = init_model(
        key, grid_shape, N=N, size_factor=size_factor, phase_type=phase_type,
        phase_kwargs=phase_kwargs, support_type=support_type,
        support_kwargs=support_kwargs,
    )
    return params


def _amplitude_from(support, Iobs, eps=1e-12):
    N = Iobs.size
    sum_I = jnp.sum(Iobs)
    sum_S = jnp.sum(support ** 2)
    return jnp.sqrt(sum_I / (N * sum_S + eps))


def forward(params, coords, Iobs, eps, model_static):
    """Single-instance forward pass.

    Returns (support, amplitude, phase_or_phasor) exactly like the original
    `PhaseRetrievalModel.call`.
    """
    support_type = model_static.get("support_type", "single")
    if support_type == "multi":
        support = compute_multi_support(
            params["support"], coords, eps, use_gates=model_static.get("use_gates", False)
        )
    else:
        support = compute_support(params["support"], coords, eps)

    amplitude = _amplitude_from(support, Iobs)

    # Bug fixed here, found by re-checking against the original phase.py:
    # the original dispatches on `hasattr(self.phaser, "compute_phasor")`,
    # and BOTH GridPhasor and DisplacementPhasor define that method (only
    # GridPhase doesn't) -- so the original ALWAYS uses the (cos, sin)
    # tuple form for 'phasor' AND 'displacement', never the raw scalar
    # phase for either. An earlier version of this file only did this for
    # 'phasor', silently falling back to the scalar form for
    # 'displacement'. The two forms give identical values in fourier_loss
    # (cos/sin either way), but they give a DIFFERENT phase-TV regularizer
    # in losses.tv_loss_phase (raw (c,s)-difference smoothness vs.
    # atan2-based wrapped-phase smoothness) -- so this only mattered when
    # beta > 0, but it is a real behavioral mismatch worth fixing.
    if model_static["phase_type"] in ("phasor", "displacement"):
        phase = compute_phasor(params["phase"], model_static)
    else:
        phase = compute_phase(params["phase"], model_static)

    return support, amplitude, phase


def loss_fn(params, static):
    """Scalar loss for one instance; this is what gets passed to the
    optimizer (see optimize.py). `static` bundles everything the optimizer
    should treat as constant.

    static = {
        "coords": (D, H, W, 3),
        "Iobs": (Do, Ho, Wo),
        "eps": float,               # half-space softness
        "alpha": float,             # support-size weight
        "beta": float,              # phase-TV weight
        "metric": "mae" | "poisson",
        "phase_static": {...},      # from init_model, incl. support_type/use_gates
    }
    """
    support, amplitude, phase = forward(
        params, static["coords"], static["Iobs"], static["eps"], static["phase_static"]
    )
    return total_loss(
        support, amplitude, phase, static["Iobs"],
        alpha=static["alpha"], beta=static["beta"], metric=static["metric"],
    )
# =============================================================================
# OPTIMIZATION
# =============================================================================
# Each population member (random restart of support + phase) is an
# independent, deterministic optimization problem sharing only Iobs -- there
# is no minibatch stochasticity here, which is exactly the regime L-BFGS is
# built for. Critically, the population members must NOT share one global
# L-BFGS step: L-BFGS's curvature history mixes information across the whole
# flattened parameter vector, so a single combined solve would spuriously
# correlate unrelated restarts' step sizes and directions. Instead we vmap an
# independent L-BFGS solve over the population axis (in_axes=0 on params,
# in_axes=None on the shared static data/config) and pick the argmin loss.
#
# We use optax.lbfgs (a real limited-memory L-BFGS with a Hessian
# approximation implicit in a short history buffer, not a dense (P, P)
# matrix -- with grid sizes up to ~6.3M voxels a dense BFGS Hessian would be
# infeasible). Peak memory added by the solver's own state is
# ~ 2 * memory_size * (#params) floats, independent of everything except
# the history depth -- this is usually the dominant per-restart memory cost
# at the largest grid sizes, more than the (now O(D*H*W)) support op or the
# FFT buffers. Reduce `memory_size` first if you need to fit more restarts.
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
import optax



def init_population(key, n_restarts, grid_shape, N=64, size_factor=4.0,
                     phase_type="grid", phase_kwargs=None,
                     support_type="single", support_kwargs=None):
    """Vmapped init of `n_restarts` independent instances.

    Returns
    -------
    params0 : pytree with a leading (n_restarts, ...) axis on every leaf.
    model_static : dict, NOT batched (it is identical across restarts by
        construction -- it depends only on grid_shape/phase_type/hkl/
        support_type/etc, never on the random key). Computed once outside
        vmap: it contains plain Python strings (phase_type, support_type)
        that vmap cannot batch.
    """
    keys = jax.random.split(key, n_restarts)

    # Static metadata does not depend on the key -- compute it once, plainly.
    _, model_static = init_model(
        keys[0], grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )

    init_one = partial(
        init_params_only, grid_shape=grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )
    params0 = jax.vmap(init_one)(keys)
    return params0, model_static


def _solve_one_lbfgs(params0, static, max_steps, tol, memory_size):
    """Single-instance L-BFGS solve (no leading batch axis)."""
    solver = optax.lbfgs(memory_size=memory_size)

    def f(p):
        return loss_fn(p, static)

    value_and_grad = optax.value_and_grad_from_state(f)

    opt_state0 = solver.init(params0)
    value0, grad0 = value_and_grad(params0, state=opt_state0)

    def cond_fn(carry):
        step, _params, _state, _value, grad = carry
        gnorm = optax.tree.norm(grad)
        return jnp.logical_and(step < max_steps, gnorm > tol)

    def body_fn(carry):
        step, params, opt_state, value, grad = carry
        updates, opt_state = solver.update(
            grad, opt_state, params, value=value, grad=grad, value_fn=f
        )
        params = optax.apply_updates(params, updates)
        value, grad = value_and_grad(params, state=opt_state)
        return (step + 1, params, opt_state, value, grad)

    init_carry = (jnp.asarray(0), params0, opt_state0, value0, grad0)
    final_step, final_params, _final_state, final_value, _final_grad = lax.while_loop(
        cond_fn, body_fn, init_carry
    )
    return final_params, final_value, final_step


def _solve_one_adam(params0, static, max_steps, tol, learning_rate):
    """Single-instance Adam solve, run for a fixed number of steps.

    ** Empirical finding, not just a theoretical concern: ** on this
    project's actual loss (MAE has an `abs()` kink; the half-space support
    has a `clip()` kink), a self-consistency test (reconstructing a target
    generated by this exact forward model, so a near-perfect fit is
    achievable in principle) showed L-BFGS's loss trace dropping fast for
    ~40 steps then STALLING around 0.085-0.10 for the remaining 260 of a
    300-step budget, despite a nonzero, non-tiny gradient norm throughout
    (0.02-0.1) -- i.e. genuine cycling/stalling from a corrupted curvature
    estimate at the kinks, not convergence. Plain Adam on the identical
    problem reached ~0.02-0.024 in the same budget, and several
    Adam-then-L-BFGS warm-start schedules were tried and all still
    underperformed plain Adam. `reconstruct(..., optimizer="adam")`
    (the default) uses this solver for that reason; L-BFGS remains
    available via `optimizer="lbfgs"` since it may still be preferable at
    tight tolerances once you are already close to a smooth local
    minimum, or on a modified loss that removes the kinks (e.g. a smooth
    L1 approximation in place of MAE's `abs()`).
    """
    solver = optax.adam(learning_rate)

    def f(p):
        return loss_fn(p, static)

    opt_state0 = solver.init(params0)
    value0, grad0 = jax.value_and_grad(f)(params0)

    def cond_fn(carry):
        step, _params, _state, _value, grad = carry
        gnorm = optax.tree.norm(grad)
        return jnp.logical_and(step < max_steps, gnorm > tol)

    def body_fn(carry):
        step, params, opt_state, value, grad = carry
        updates, opt_state = solver.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        value, grad = jax.value_and_grad(f)(params)
        return (step + 1, params, opt_state, value, grad)

    init_carry = (jnp.asarray(0), params0, opt_state0, value0, grad0)
    final_step, final_params, _final_state, final_value, _final_grad = lax.while_loop(
        cond_fn, body_fn, init_carry
    )
    return final_params, final_value, final_step


class ReconstructionResult(NamedTuple):
    best_params: dict          # single-instance pytree (argmin over restarts)
    best_loss: jnp.ndarray     # scalar
    all_losses: jnp.ndarray    # (n_restarts,)
    all_steps: jnp.ndarray     # (n_restarts,)
    coords: jnp.ndarray        # (D, H, W, 3), shared
    model_static: dict         # shared; phase_type/support_type/etc.
    eps: float

    def evaluate(self, Iobs):
        """Recompute (support, amplitude, phase) for the best restart."""
        return forward(self.best_params, self.coords, Iobs, self.eps, self.model_static)


def reconstruct(
    key,
    Iobs,
    n_restarts,
    N=64,
    size_factor=4.0,
    eps=0.6,
    alpha=0.8,
    beta=0.1,
    metric="mae",
    phase_type="grid",
    phase_kwargs=None,
    support_type="single",
    support_kwargs=None,
    optimizer="adam",
    max_steps=300,
    tol=1e-6,
    memory_size=10,
    learning_rate=0.05,
    grid_shape=None,
):
    """Run `n_restarts` independent optimizations in parallel (vmapped) and
    return the best one by final loss.

    Parameters
    ----------
    support_type : "single" (default) | "multi"
        "multi" reconstructs a non-convex object as a soft union of several
        convex parts (see multi_support.py). `support_kwargs` may contain
        "M_parts" (default 2), "N_per_part" (default N), "use_gates"
        (default False).
    optimizer : "adam" (default) | "lbfgs"
        See `_solve_one_adam`'s docstring for why "adam" is the default:
        on this project's actual (non-smooth) loss, L-BFGS was observed to
        stall well short of a good minimum in a self-consistency test,
        while Adam did not. Use "lbfgs" if you've modified the loss to
        remove its kinks, or want to fine-tune the last mile after an
        Adam run of your own.
    learning_rate : float
        Only used when `optimizer="adam"`.
    memory_size : int
        Only used when `optimizer="lbfgs"`; L-BFGS history depth.

    Notes
    -----
    - `n_restarts` is capped in practice by GPU memory. For "lbfgs" this is
      dominated by the L-BFGS history (~ 2 * memory_size * n_params floats
      per restart) at large grid sizes; lower `memory_size` to fit more
      restarts. For "adam" the optimizer state is O(n_params) regardless.
    - Because this uses `lax.while_loop` under `vmap`, all restarts run in
      lockstep: the loop only exits once every lane has met its stopping
      criterion. Restarts that converge early just perform cheap no-op-ish
      steps afterwards -- correctness is unaffected, only wall-clock cost.
    """
    Iobs = jnp.asarray(Iobs, dtype=jnp.float32)
    if grid_shape is None:
        grid_shape, coords = make_coords_for(Iobs.shape)
    else:
        from .support import make_coords
        coords = make_coords(grid_shape)

    params0, model_static = init_population(
        key, n_restarts, grid_shape, N=N, size_factor=size_factor,
        phase_type=phase_type, phase_kwargs=phase_kwargs,
        support_type=support_type, support_kwargs=support_kwargs,
    )

    static = {
        "coords": coords,
        "Iobs": Iobs,
        "eps": eps,
        "alpha": alpha,
        "beta": beta,
        "metric": metric,
        "phase_static": model_static,
    }

    if optimizer == "adam":
        solve = partial(_solve_one_adam, static=static, max_steps=max_steps,
                         tol=tol, learning_rate=learning_rate)
    elif optimizer == "lbfgs":
        solve = partial(_solve_one_lbfgs, static=static, max_steps=max_steps,
                         tol=tol, memory_size=memory_size)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}, choose 'adam' or 'lbfgs'.")

    batched_solve = jax.vmap(solve, in_axes=(0,))

    final_params, final_values, final_steps = batched_solve(params0)

    best_idx = jnp.argmin(final_values)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], final_params)

    return ReconstructionResult(
        best_params=best_params,
        best_loss=final_values[best_idx],
        all_losses=final_values,
        all_steps=final_steps,
        coords=coords,
        model_static=model_static,
        eps=eps,
    )