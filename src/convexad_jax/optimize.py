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

from .model import init_model, init_params_only, make_coords_for, loss_fn, forward
from .support_freeform import init_freeform_support_params, invert_support_to_logit


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


def _solve_one_adam(
    params0, static, max_steps, tol, learning_rate,
    decay_steps=500, decay_rate=0.9, staircase=True,
    b1=0.9, b2=0.98, eps_adam=1e-6,
):
    """Single-instance Adam(AMSGrad) solve with LR decay, run for a fixed
    number of steps (or until gradient norm drops below `tol`).

    ** Empirical finding, not just a theoretical concern: ** on this
    project's actual loss (MAE has an `abs()` kink; the half-space support
    has a `clip()` kink), a self-consistency test ... [unchanged docstring]
    """
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )
    # optax.adam has no `amsgrad` flag directly -- chain the AMSGrad
    # second-moment rule with the same LR schedule TF used.
    solver = optax.chain(
        optax.scale_by_amsgrad(b1=b1, b2=b2, eps=eps_adam),
        optax.scale_by_learning_rate(schedule),
    )

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
    all_params: Optional[dict] = None
    # Full population pytree (leading (n_restarts, ...) axis on every
    # leaf), not just the argmin. Cheap to keep for "single"/"multi"
    # support (O(N) or O(M*N) params/restart); can be sizeable for
    # "freeform" (O(D*H*W) params/restart) at large grids/n_restarts --
    # this is what `reconstruct_two_stage` uses to pick more than the
    # single best stage-1 restart to carry into stage 2.

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
    max_steps=5000,          # was 300 -- TF ran 5000; cheap under while_loop/vmap
    tol=1e-6,
    memory_size=10,
    learning_rate=0.05,
    decay_steps=500,         # NEW -- matches TF's ExponentialDecay
    decay_rate=0.9,          # NEW
    staircase=True,          # NEW
    b1=0.9,                  # NEW
    b2=0.98,                 # NEW -- matches TF (optax default is 0.999)
    eps_adam=1e-6,           # NEW -- matches TF (optax default is 1e-8)
    grid_shape=None,
):

    

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
        solve = partial(
            _solve_one_adam, static=static, max_steps=max_steps, tol=tol,
            learning_rate=learning_rate, decay_steps=decay_steps,
            decay_rate=decay_rate, staircase=staircase,
            b1=b1, b2=b2, eps_adam=eps_adam,
        )
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
        all_params=final_params,
    )


# =============================================================================
# TWO-STAGE (CONVEX -> FREE-FORM) RECONSTRUCTION
# =============================================================================
# Stage 1 is a normal `reconstruct(..., support_type="single" | "multi")`
# population solve: cheap (O(N) or O(M*N) support params/restart), and its
# job is only to find the rough global shape and phase field, robustly,
# via the usual multi-restart search.
#
# Stage 2 releases a SMALL number of the best stage-1 restarts to the
# free-form (support_freeform.py) voxel-grid support and refines them,
# each anchored to its own stage-1 shape via a T (sharpness) / zeta
# (anchor weight) continuation schedule that starts tight (trust the
# stage-1 shape) and relaxes (let genuinely non-convex detail -- a twin
# facet, a second particle, a concave notch -- emerge). See
# support_freeform.py's module docstring for why a cold free-form start is
# a bad idea.

def _solve_one_adam_freeform(
    params0, S_ref, base_static, stage_schedule, tol,
    learning_rate, decay_steps, decay_rate, staircase, b1, b2, eps_adam,
):
    """Single-instance staged Adam solve for the free-form support.

    Unlike `_solve_one_adam`, `S_ref` (this restart's anchor target -- the
    stage-1 support it was warm-started from) is an explicit argument
    rather than folded into `base_static`, specifically so it can vary per
    population member under `vmap(..., in_axes=(0, 0))` while
    `base_static` (Iobs, coords, alpha, beta, gamma, delta, metric,
    phase_static) stays shared (`in_axes=None`). This is what lets several
    kept stage-1 restarts be released to free-form in parallel, each
    anchored to its own stage-1 solution, instead of all sharing one.

    `stage_schedule` is a small Python-level (static, not traced) sequence
    of (T, zeta, max_steps) triples -- a continuation schedule for the
    support sharpness T (passed through as `static["eps"]`, reusing the
    existing eps-as-softness convention from support.py/multi_support.py)
    and the anchor weight zeta. Each stage is one `_solve_one_adam`-style
    `lax.while_loop`; the stages are unrolled in plain Python (there are
    only ever a handful), so this still traces to a single jaxpr per
    restart under vmap/jit, and gets a fresh Adam state at each stage
    boundary (a small, deliberate reset -- the loss landscape genuinely
    changes shape each time T or zeta changes, so stale second-moment
    estimates from the previous stage aren't worth carrying over).
    """
    params = params0
    final_value = jnp.asarray(jnp.inf, dtype=jnp.float32)
    total_steps = jnp.asarray(0)

    for T_i, zeta_i, steps_i in stage_schedule:
        stage_static = dict(base_static)
        stage_static["eps"] = T_i
        stage_static["zeta"] = zeta_i
        stage_static["S_ref"] = S_ref
        params, final_value, steps = _solve_one_adam(
            params, stage_static, max_steps=steps_i, tol=tol,
            learning_rate=learning_rate, decay_steps=decay_steps,
            decay_rate=decay_rate, staircase=staircase,
            b1=b1, b2=b2, eps_adam=eps_adam,
        )
        total_steps = total_steps + steps

    return params, final_value, total_steps


def reconstruct_two_stage(
    key,
    Iobs,
    n_restarts_stage1,
    stage1_support_type="single",
    n_keep=1,
    n_restarts_stage2=None,
    grid_shape=None,
    stage1_kwargs=None,
    T_schedule=(1.5, 0.6, 0.25),
    zeta_schedule=(3.0, 1.0, 0.0),
    steps_per_stage=1000,
    gamma=1e-3,
    delta=1e-2,
    noise_scale_support=0.0,
    noise_scale_phase=0.0,
    stage2_learning_rate=0.02,
    stage2_decay_steps=500,
    stage2_decay_rate=0.9,
    stage2_staircase=True,
    stage2_b1=0.9,
    stage2_b2=0.98,
    stage2_eps_adam=1e-6,
    tol=1e-6,
):
    """Two-stage reconstruction for non-convex / multi-particle supports:
    a cheap convex/multi-convex stage 1, then a free-form release for a
    small population of the best stage-1 candidates.

    Parameters
    ----------
    n_restarts_stage1, stage1_support_type : population size and support
        parameterization ("single" or "multi") for the stage-1 solve --
        forwarded to `reconstruct` via `stage1_kwargs`.
    n_keep : how many of the lowest-loss stage-1 restarts to carry into
        stage 2. Keep this small (1-3): each one becomes an O(D*H*W)-param
        free-form restart, a very different memory regime than stage 1
        (see `support_freeform.py`'s and `support.py`'s module docstrings
        on per-restart memory cost).
    n_restarts_stage2 : population size for stage 2; defaults to `n_keep`
        (one free-form restart per kept candidate). If larger, extra
        restarts are assigned to kept candidates round-robin and
        decorrelated via `noise_scale_support`/`noise_scale_phase`.
    stage1_kwargs : dict forwarded to `reconstruct` for stage 1 (N,
        size_factor, eps, alpha, beta, metric, phase_type, phase_kwargs,
        support_kwargs, optimizer, max_steps, tol, memory_size,
        learning_rate, ...). `support_type` is set from
        `stage1_support_type` if not already present.
    T_schedule, zeta_schedule : continuation schedule for stage 2 -- same
        length, T decreasing (support sharpness, see
        `support_freeform.compute_freeform_support`) and zeta decreasing
        toward 0 (anchor weight toward the stage-1 shape, see
        `support_freeform.anchor_penalty`). Defaults are a reasonable
        starting point, not tuned for any particular dataset.
    steps_per_stage : Adam steps (or until `tol`) per schedule stage.
    gamma, delta : perimeter / double-well regularizer weights (see
        `support_freeform.tv_support` / `double_well_support`), held
        constant across all stage-2 stages.
    noise_scale_support, noise_scale_phase : stddev of Gaussian noise added
        to the support logit / phase params of each stage-2 restart on top
        of its assigned stage-1 warm start -- only matters when
        `n_restarts_stage2 > n_keep`, to decorrelate restarts that would
        otherwise be exact duplicates.

    Returns
    -------
    stage1_result, stage2_result : both `ReconstructionResult`. On
    `stage2_result`, `eps` is set to `T_schedule[-1]` (the T needed to
    reproduce the final support via `.evaluate()`), and `model_static`
    has `support_type` overridden to `"freeform"`.
    """
    if len(T_schedule) != len(zeta_schedule):
        raise ValueError("T_schedule and zeta_schedule must have the same length")

    key1, key2 = jax.random.split(key)
    stage1_kwargs = dict(stage1_kwargs or {})
    stage1_kwargs.setdefault("support_type", stage1_support_type)
    alpha = stage1_kwargs.get("alpha", 0.8)
    beta = stage1_kwargs.get("beta", 0.1)
    metric = stage1_kwargs.get("metric", "mae")

    stage1_result = reconstruct(
        key1, Iobs, n_restarts=n_restarts_stage1, grid_shape=grid_shape,
        **stage1_kwargs,
    )

    Iobs_arr = jnp.asarray(Iobs, dtype=jnp.float32)
    coords = stage1_result.coords
    grid_shape = coords.shape[:3]

    n_keep = min(n_keep, n_restarts_stage1)
    keep_idx = jnp.argsort(stage1_result.all_losses)[:n_keep]
    kept_params = jax.tree_util.tree_map(lambda x: x[keep_idx], stage1_result.all_params)

    # Re-evaluate the kept restarts' converged support at stage 1's own
    # (fixed) eps -- this is the shape stage 2 warm-starts from and
    # anchors to.
    kept_forward = jax.vmap(
        lambda p: forward(p, coords, Iobs_arr, stage1_result.eps, stage1_result.model_static)
    )
    kept_support, _kept_amplitude, _kept_phase = kept_forward(kept_params)  # (n_keep, D,H,W)
    kept_logit = jax.vmap(invert_support_to_logit)(kept_support)            # (n_keep, D,H,W)

    n_restarts_stage2 = n_restarts_stage2 or n_keep
    assign = jnp.arange(n_restarts_stage2) % n_keep  # round-robin over kept candidates
    logits0 = kept_logit[assign]
    phase0 = jax.tree_util.tree_map(lambda x: x[assign], kept_params["phase"])
    S_ref_batched = kept_support[assign]                                    # (n_restarts_stage2, D,H,W)

    def _init_stage2_one(k, logit0, phase_p0):
        k_support, k_phase = jax.random.split(k)
        support_params = init_freeform_support_params(
            k_support, grid_shape, init_logit=logit0, noise_scale=noise_scale_support
        )
        if noise_scale_phase > 0.0:
            phase_params = jax.tree_util.tree_map(
                lambda x: x + noise_scale_phase * jax.random.normal(k_phase, x.shape),
                phase_p0,
            )
        else:
            phase_params = phase_p0
        return {"support": support_params, "phase": phase_params}

    keys2 = jax.random.split(key2, n_restarts_stage2)
    params0_stage2 = jax.vmap(_init_stage2_one)(keys2, logits0, phase0)

    model_static_stage2 = dict(stage1_result.model_static)
    model_static_stage2["support_type"] = "freeform"

    base_static = {
        "coords": coords,
        "Iobs": Iobs_arr,
        "alpha": alpha,
        "beta": beta,
        "metric": metric,
        "gamma": gamma,
        "delta": delta,
        "phase_static": model_static_stage2,
    }
    stage_schedule = tuple(
        (T_i, zeta_i, steps_per_stage) for T_i, zeta_i in zip(T_schedule, zeta_schedule)
    )

    solve = partial(
        _solve_one_adam_freeform, base_static=base_static, stage_schedule=stage_schedule,
        tol=tol, learning_rate=stage2_learning_rate, decay_steps=stage2_decay_steps,
        decay_rate=stage2_decay_rate, staircase=stage2_staircase,
        b1=stage2_b1, b2=stage2_b2, eps_adam=stage2_eps_adam,
    )
    batched_solve = jax.vmap(solve, in_axes=(0, 0))
    final_params, final_values, final_steps = batched_solve(params0_stage2, S_ref_batched)

    best_idx = jnp.argmin(final_values)
    best_params = jax.tree_util.tree_map(lambda x: x[best_idx], final_params)

    stage2_result = ReconstructionResult(
        best_params=best_params,
        best_loss=final_values[best_idx],
        all_losses=final_values,
        all_steps=final_steps,
        coords=coords,
        model_static=model_static_stage2,
        eps=T_schedule[-1],
        all_params=final_params,
    )
    return stage1_result, stage2_result
