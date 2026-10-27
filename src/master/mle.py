import jax.numpy as jnp
import numpy as np
from master.fft import *
from jax import jit
from jax.scipy.sparse.linalg import cg
from jax import lax


@jit
def log_likelihood_x(d, x, A=fft2_shifted, epsilon=1e-8):
    return log_likelihood(d, A(x), epsilon)


@jit
def log_likelihood_pty(d, y_model, epsilon=1e-8):
    return np.sum(log_likelihood(d, y_model, epsilon), axis=1)


@jit
def log_likelihood(d, y_model, epsilon=1e-8):
    """Log likelihood of independent Poisson model modulo constant factors"""
    mu_model = jnp.abs(y_model) ** 2
    return jnp.sum(d * jnp.log(mu_model + epsilon) - mu_model, axis=[-2, -1])


def calc_ell_init(d, init_est, N_samples, A, probes, chunk=16):
    ell_init = np.zeros((N_samples))
    for i in range(0, N_samples // chunk):
        start = i * chunk
        end = (i + 1) * chunk
        # Use batching along sample axis
        ys = A(init_est[start:end, None, :, :] * probes[None, :, :, :])
        ell_ys = log_likelihood(d[start:end, :, :, :], ys)
        ell_init[start:end] = np.sum(ell_ys, axis=1)
    return ell_init


def norm_ll(ll):
    ll_max = np.max(ll, axis=0)
    ll_min = np.min(ll, axis=0)
    return (ll - ll_min) / (ll_max - ll_min)


def construct_norm_ll(ll):
    ll_max = np.max(ll, axis=0)
    ll_min = np.min(ll, axis=0)
    return lambda x: (x - ll_min) / (ll_max - ll_min)


def ll_flag_to_f_track(track_ll):
    if track_ll:
        f_track = log_likelihood
    else:
        f_track = None
    return f_track


def step_linear(x_model, g, **kwargs):
    return g


def step_fisher_solve(x_model, g, cg_max_iter, cg_tol, **kwargs):
    # Use a CG-based solver to solve Jx_FS = g
    return solve_fisher(x_model, g, cg_max_iter=cg_max_iter, cg_tol=cg_tol)


@jit(static_argnames=("A", "A_dagger", "f_track", "N_iter", "f_step"))
def ml_refinement_generic_tracking(
    x_initial,
    x_data,
    obj_support,
    alpha=2e-3,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    epsilon=1e-12,
    N_iter=10000,
    f_track=None,
    f_step=step_linear,
    **kwargs,
):
    """Refines an initial reconstruction estimate x_initial to a maximum likelihood solution using gradient ascent on the log likelihood. This maximizes the likelihood of the data as a function of model x."""
    conv_count = x_initial.shape[:-2]
    track_vals = np.zeros((N_iter, *conv_count))
    d = jnp.abs(x_data) ** 2

    def update_step(it, v):
        x_model, track_vals = v
        y_model = A(x_model)
        if f_track is not None:
            track_val = f_track(d, y_model)
            track_vals = track_vals.at[it].set(track_val)
        # Gradient to x^* of log likelihood for linear complex operator A
        g = A_dagger((d / (jnp.abs(y_model) ** 2 + epsilon) - 1) * y_model)
        # TODO: factor out obj_support and make object-space constraint a more generic projection
        x_model = (x_model + 0.5 * alpha * f_step(x_model, g, **kwargs)) * obj_support
        return x_model, track_vals

    x_model, track_vals = lax.fori_loop(0, N_iter, update_step, (x_initial, track_vals))
    return x_model, track_vals


@jit(static_argnames=("A", "A_dagger", "track_ll", "N_iter"))
def ml_refinement_first_order(
    x_initial,
    x_data,
    obj_support,
    alpha=2e-3,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    epsilon=1e-12,
    N_iter=10000,
    track_ll=False,
):
    return ml_refinement_generic_tracking(
        x_initial,
        x_data,
        obj_support,
        alpha,
        A,
        A_dagger,
        epsilon,
        N_iter,
        ll_flag_to_f_track(track_ll),
    )


@jit(static_argnames=("A", "A_dagger", "track_ll", "N_iter"))
def ml_fisher_scoring_unitary_A(
    x_initial,
    x_data,
    obj_support,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    epsilon=1e-12,
    N_iter=10000,
    alpha=1.0,
    track_ll=False,
):
    """Assuming unitary A, Fisher scoring is the first order method with alpha = 1.0."""
    return ml_refinement_first_order(
        x_initial,
        x_data,
        obj_support,
        alpha=alpha,
        A=A,
        A_dagger=A_dagger,
        epsilon=epsilon,
        N_iter=N_iter,
        track_ll=track_ll,
    )


@jit(static_argnames=("A", "A_dagger"))
def fisher_op(delta_x, x, A=ifft2_shifted, A_dagger=fft2_shifted, eps=1e-12):
    y = A(x)
    mu = jnp.abs(y) ** 2 + eps

    u = A(delta_x)

    proj = jnp.real(jnp.conj(y) * u) / mu
    proj = proj * y

    return A_dagger(proj)


@jit(static_argnames=("A", "A_dagger"))
def fisher_op_pseudoinverse(
    delta_x, x, A=fft2_shifted, A_dagger=ifft2_shifted, eps=1e-8
):
    y = A(x)
    mu = jnp.abs(y) ** 2 + eps

    u = A(delta_x)
    return 1 / 4 * A_dagger((jnp.real(jnp.conj(y) * u) / mu) * y)


@jit(static_argnames=("A", "A_dagger"))
def solve_fisher(
    x,
    g,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    eps=1e-12,
    cg_max_iter=None,
    cg_tol=1e-5,
):
    @jit
    def matvec(v):
        return fisher_op(v, x, A, A_dagger, eps)

    delta_x, _ = cg(matvec, g, maxiter=cg_max_iter, tol=cg_tol)
    return delta_x


@jit(static_argnames=("A", "A_dagger", "track_ll", "N_iter"))
def ml_fisher_scoring_general_A(
    x_initial,
    x_data,
    obj_support,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    epsilon=1e-12,
    N_iter=250,
    alpha=1.0,
    track_ll=False,
    cg_max_iter=None,
    cg_tol=1e-5,
):
    return ml_refinement_generic_tracking(
        x_initial,
        x_data,
        obj_support,
        alpha,
        A,
        A_dagger,
        epsilon,
        N_iter,
        ll_flag_to_f_track(track_ll),
        f_step=step_fisher_solve,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
    )
