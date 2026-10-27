from jax import lax
from master.fft import *
import jax.numpy as jnp
from master.mle import log_likelihood


def ptycho_consensus(z, probes, obj_support, epsilon=1e-12):
    """Find the least-squares object-space consensus estimate from exit waves."""
    return (
        jnp.sum(jnp.conj(probes) * z, axis=-3)
        / (jnp.sum(jnp.abs(probes) ** 2, axis=-3) + epsilon)
        * obj_support
    )


@jit(static_argnames=("track_ll", "N_iter", "A", "A_dagger"))
def mle_ptycho_first_order(
    x_initial,
    probes,
    x_data,
    obj_support,
    alpha=1e-5,
    epsilon=1e-12,
    N_iter=1000,
    track_ll=False,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    """Refines an initial reconstruction estimate x_initial to a maximum likelihood solution using gradient ascent on the log likelihood. This maximizes the likelihood of the data as a function of model x."""
    conv_count = x_initial.shape[0]
    lls_first_order = jnp.zeros((N_iter, conv_count))
    x_model_ml_first_order = x_initial * obj_support
    d = jnp.abs(x_data) ** 2
    # for it in tqdm(range(N_iter)):

    def update_step(it, v):
        x_model_ml_first_order, lls_first_order = v
        ys_model = A(x_model_ml_first_order[:, None, :, :] * probes[None, :, :, :])
        if track_ll:
            lls = log_likelihood(d, ys_model)
            print(lls.shape)
            lls_first_order = lls_first_order.at[it].set(jnp.sum(lls, axis=1))
        g = jnp.sum(
            jnp.conj(probes)
            * A_dagger((d / (jnp.abs(ys_model) ** 2 + epsilon) - 1) * ys_model),
            axis=-3,
        )
        # Basic first order gradient descent
        # TODO: factor out obj_support and make object-space constraint a more generic projection
        x_model_ml_first_order = x_model_ml_first_order + alpha * g * obj_support

        return x_model_ml_first_order, lls_first_order

    x_model_ml_first_order, lls_first_order = lax.fori_loop(
        0, N_iter, update_step, (x_model_ml_first_order, lls_first_order)
    )

    return x_model_ml_first_order, lls_first_order


@jit(static_argnames=("track_ll", "N_iter", "A", "A_dagger"))
def mle_ptycho_first_order_consensus(
    x_initial,
    probes,
    x_data,
    obj_support,
    alpha=1e-3,
    epsilon=1e-12,
    N_iter=100,
    track_ll=False,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    """First-order per-frame ML updates followed by object-space consensus."""
    conv_count = x_initial.shape[0]
    ll_tracks = jnp.zeros((N_iter, conv_count))
    x_model_ml = x_initial * obj_support
    d = jnp.abs(x_data) ** 2

    def update_step(it, v):
        x_model_ml, ll_tracks = v

        z_model_ml = x_model_ml[:, None, :, :] * probes[None, :, :, :]
        ys_model = A(z_model_ml)

        if track_ll:
            lls = log_likelihood(d, ys_model)
            ll_tracks = ll_tracks.at[it].set(jnp.sum(lls, axis=1))

        g = A_dagger((d / (jnp.abs(ys_model) ** 2 + epsilon) - 1) * ys_model)

        z_model_ml = z_model_ml + alpha * g / 2

        x_model_ml = ptycho_consensus(z_model_ml, probes, obj_support, epsilon)

        return x_model_ml, ll_tracks

    x_model_ml, ll_tracks = lax.fori_loop(
        0, N_iter, update_step, (x_model_ml, ll_tracks)
    )

    return x_model_ml, ll_tracks


@jit(static_argnames=("track_ll", "N_iter", "A", "A_dagger"))
def mle_ptycho_second_order_consensus(
    x_initial,
    probes,
    x_data,
    obj_support,
    alpha=1.0,
    epsilon=1e-12,
    N_iter=100,
    track_ll=False,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    """Refines an initial reconstruction estimate using per-frame Fisher scoring
    updates followed by least-squares consensus in object space. A is required
    to be unitary for the used identity to work."""
    return mle_ptycho_first_order_consensus(
        x_initial,
        probes,
        x_data,
        obj_support,
        alpha=alpha,
        epsilon=epsilon,
        N_iter=N_iter,
        track_ll=track_ll,
        A=A,
        A_dagger=A_dagger,
    )
