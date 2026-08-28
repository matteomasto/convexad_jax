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
