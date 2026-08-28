"""Gradient-correctness and integration tests.

These are the checks used during development to find and fix real bugs in
this port (see README.md's "Bugs found and fixed" section) -- kept here so
they run in CI on every change, rather than as one-off scripts.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from convex_ad_jax.support import (
    make_coords, halfspace_support, init_support_params, compute_support,
    stereographic_to_unit, unit_to_stereographic,
)
from convex_ad_jax.multi_support import init_multi_support_params, compute_multi_support
from convex_ad_jax.model import init_model, forward, loss_fn, make_coords_for
from convex_ad_jax.optimize import reconstruct, init_population
from convex_ad_jax.utils import center_pad, project, save_params_npz, load_params_npz


def test_stereographic_round_trip():
    key = jax.random.PRNGKey(0)
    n0 = jax.random.normal(key, (8, 3))
    n0 = n0 / jnp.linalg.norm(n0, axis=-1, keepdims=True)
    p0 = unit_to_stereographic(n0)
    n0_rt = stereographic_to_unit(p0)
    assert float(jnp.max(jnp.abs(n0 - n0_rt))) < 1e-5


def test_halfspace_support_gradient_matches_finite_difference():
    grid_shape = (10, 9, 8)
    coords = make_coords(grid_shape)
    N = 5
    key = jax.random.PRNGKey(0)
    sp = init_support_params(key, N, grid_shape)

    def loss(sp):
        S = compute_support(sp, coords, 0.6)
        return jnp.sum(S ** 2)

    g = jax.grad(loss)(sp)
    h = 1e-2
    i0, j0 = 2, 1
    plus = {**sp, "p_raw": sp["p_raw"].at[i0, j0].add(h)}
    minus = {**sp, "p_raw": sp["p_raw"].at[i0, j0].add(-h)}
    fd = (loss(plus) - loss(minus)) / (2 * h)
    rel_err = abs(float(g["p_raw"][i0, j0]) - float(fd)) / (abs(float(fd)) + 1e-12)
    assert rel_err < 0.05


def test_clip_boundary_gradient_is_masked():
    """Regression test for the clip-boundary gradient bug: a plane held
    deep in saturation everywhere must contribute exactly zero gradient."""
    jax.config.update("jax_enable_x64", True)
    try:
        grid_shape = (12, 10, 8)
        coords = make_coords(grid_shape).astype(jnp.float64)
        key = jax.random.PRNGKey(1)
        n = jax.random.normal(key, (3, 3)).astype(jnp.float64)
        n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
        d = jnp.array([-100.0, 50.0, 50.0], dtype=jnp.float64)

        def loss_sum_S(d0):
            d_full = d.at[0].set(d0)
            S = halfspace_support(n, d_full, coords, 0.6)
            return jnp.sum(S)

        g = jax.grad(loss_sum_S)(d[0])
        assert abs(float(g)) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", False)


def test_multi_support_centers_gradient_flows():
    """Regression test: centers must receive a nonzero, correct gradient
    (the custom VJP used to hardcode None for d(support)/d(coords))."""
    jax.config.update("jax_enable_x64", True)
    try:
        grid_shape = (14, 12, 10)
        coords = make_coords(grid_shape).astype(jnp.float64)
        key = jax.random.PRNGKey(0)
        params = init_multi_support_params(key, 3, 4, grid_shape, use_gates=True)
        params = {k: v.astype(jnp.float64) for k, v in params.items()}

        def loss(p):
            S = compute_multi_support(p, coords, 0.6, use_gates=True)
            return jnp.sum(S ** 2)

        g = jax.grad(loss)(params)
        h = 1e-4
        idx = (0, 2)
        plus = dict(params); plus["centers"] = params["centers"].at[idx].add(h)
        minus = dict(params); minus["centers"] = params["centers"].at[idx].add(-h)
        fd = (loss(plus) - loss(minus)) / (2 * h)
        assert abs(float(g["centers"][idx]) - float(fd)) < 1e-6
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("phase_type,kwargs", [
    ("grid", None), ("phasor", None), ("displacement", {"hkl": (1, 1, 1)}),
])
def test_forward_and_loss_all_phase_types(phase_type, kwargs):
    Iobs = np.random.default_rng(3).random((12, 10, 8)).astype(np.float32)
    grid_shape, coords = make_coords_for(Iobs.shape)
    key = jax.random.PRNGKey(0)
    params, model_static = init_model(key, grid_shape, N=4, phase_type=phase_type,
                                       phase_kwargs=kwargs)
    support, amplitude, phase = forward(params, coords, jnp.asarray(Iobs), 0.6, model_static)
    assert isinstance(phase, tuple) == (phase_type != "grid")

    static = {"coords": coords, "Iobs": jnp.asarray(Iobs), "eps": 0.6, "alpha": 0.0,
              "beta": 0.0, "metric": "mae", "phase_static": model_static}
    loss = loss_fn(params, static)
    grad = jax.grad(loss_fn)(params, static)
    assert jnp.isfinite(loss)


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_reconstruct_single_support(optimizer):
    Iobs = np.random.default_rng(2).random((12, 10, 8)).astype(np.float32)
    key = jax.random.PRNGKey(0)
    res = reconstruct(key, Iobs, n_restarts=2, N=4, eps=0.6, max_steps=5, tol=1e-8,
                       optimizer=optimizer)
    assert res.all_losses.shape == (2,)
    assert jnp.isfinite(res.best_loss)


def test_reconstruct_multi_support():
    Iobs = np.random.default_rng(2).random((16, 14, 12)).astype(np.float32)
    key = jax.random.PRNGKey(0)
    res = reconstruct(key, Iobs, n_restarts=2, N=4, eps=0.6, max_steps=5, tol=1e-8,
                       support_type="multi",
                       support_kwargs={"M_parts": 2, "N_per_part": 4})
    assert "centers" in res.best_params["support"]
    assert jnp.isfinite(res.best_loss)


def test_save_load_params_roundtrip(tmp_path):
    Iobs = np.random.default_rng(0).random((12, 10, 8)).astype(np.float32)
    key = jax.random.PRNGKey(0)
    res = reconstruct(key, Iobs, n_restarts=2, N=4, eps=0.6, max_steps=3, tol=1e-8)

    path = tmp_path / "params.npz"
    save_params_npz(res.best_params, str(path))
    loaded = load_params_npz(str(path))
    for a, b in zip(jax.tree_util.tree_leaves(res.best_params),
                     jax.tree_util.tree_leaves(loaded)):
        assert np.allclose(np.asarray(a), np.asarray(b))


def test_project_shapes():
    Iobs = np.random.default_rng(0).random((12, 10, 8)).astype(np.float32)
    obj = np.random.default_rng(1).normal(size=(6, 5, 4)) + 1j * np.random.default_rng(2).normal(size=(6, 5, 4))
    obj_p = center_pad(jnp.asarray(obj), Iobs.shape)
    refined = project(obj_p, jnp.asarray(Iobs))
    assert refined.shape == Iobs.shape
