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
    )
