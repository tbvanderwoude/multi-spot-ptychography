import jax.numpy as jnp
from master.fft import *
import jax
from jax import lax
from .phase_retrieval import construct_random_phi


@jit(static_argnames=("A", "A_dagger"))
def pie_measurement_update_guard(
    x_est,
    y_m,
    probe_m,
    alpha,
    gamma,
    epsilon,
    beta_reg,
    obj_support,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    z_est = x_est * probe_m
    y_est = A(z_est)
    y_abs = jnp.abs(y_est)
    y_corr = jnp.where(
        y_abs > 0, jnp.abs(y_m) * y_est / jnp.where(y_abs > 0, y_abs, 1.0), jnp.abs(y_m)
    )

    # Increasing beta adds a type of regularization that drives down the part of the object*probe
    # fields that produces no change in y, such as frequencies outside of the synthesized aperture
    delta_z = A_dagger(y_corr + (beta_reg - 1.0) * y_est) - beta_reg * z_est

    probe_int = jnp.abs(probe_m) ** 2
    denom = (1.0 - alpha) * probe_int + alpha * jnp.max(probe_int)

    x_est = (x_est + gamma * jnp.conj(probe_m) / denom * delta_z) * obj_support

    return x_est


@jit(static_argnames=("A", "A_dagger"))
def pie_measurement_update(
    x_est,
    y_m,
    probe_m,
    alpha,
    gamma,
    epsilon,
    beta_reg,
    obj_support,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    z_est = x_est * probe_m
    y_est = A(z_est)
    y_corr = jnp.abs(y_m) * y_est / (jnp.abs(y_est) + epsilon)

    # Increasing beta adds a type of regularization that drives down the part of the object*probe
    # fields that produces no change in y, such as frequencies outside of the synthesized aperture
    delta_z = A_dagger(y_corr + (beta_reg - 1.0) * y_est) - beta_reg * z_est

    probe_int = jnp.abs(probe_m) ** 2
    denom = (1.0 - alpha) * probe_int + alpha * jnp.max(probe_int) + epsilon

    x_est = (x_est + gamma * jnp.conj(probe_m) / denom * delta_z) * obj_support

    return x_est


@jit(static_argnames=("A", "A_dagger", "use_flat_init"))
def pie_spectral_init(
    ys,
    probes,
    base_key=jax.random.PRNGKey(1776),
    alpha=0.9,
    gamma=0.75,
    epsilon=1e-8,
    beta_reg=0.1,
    N_iter=1000,
    obj_support=1,
    obj_support_init=1,
    spectral_support=1,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
    use_flat_init=False,
):
    if use_flat_init:
        x_est = lax.complex(obj_support_init, 0.0)
    else:
        x_est = ifft2_shifted(
            spectral_support
            * jnp.exp(1j * construct_random_phi(base_key, 0, probes[0].shape))
        )
    M = probes.shape[0]

    def iteration_step(_, carry):
        x_est, key = carry
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, M)

        def update_step(j, x_est):
            m = indices[j]
            return pie_measurement_update(
                x_est,
                ys[m],
                probes[m],
                alpha,
                gamma,
                epsilon,
                beta_reg,
                obj_support,
                A,
                A_dagger,
            )

        x_est = lax.fori_loop(0, M, update_step, x_est)
        return x_est, key

    x_est, _ = lax.fori_loop(0, N_iter, iteration_step, (x_est, base_key))
    return x_est


@jit(static_argnames=("A", "A_dagger"))
def pie(
    ys,
    probes,
    base_key=jax.random.PRNGKey(1776),
    alpha=0.9,
    gamma=0.75,
    epsilon=1e-8,
    beta_reg=0.1,
    N_iter=1000,
    obj_support=1,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    x_est = obj_support * jnp.exp(
        1j * construct_random_phi(base_key, 0, probes[0].shape)
    )
    M = probes.shape[0]

    def iteration_step(_, carry):
        x_est, key = carry
        key, subkey = jax.random.split(key)
        indices = jax.random.permutation(subkey, M)

        def update_step(j, x_est):
            m = indices[j]
            return pie_measurement_update(
                x_est,
                ys[m],
                probes[m],
                alpha,
                gamma,
                epsilon,
                beta_reg,
                obj_support,
                A,
                A_dagger,
            )

        x_est = lax.fori_loop(0, M, update_step, x_est)
        return x_est, key

    x_est, _ = lax.fori_loop(0, N_iter, iteration_step, (x_est, base_key))
    return x_est


@jit(static_argnames=("A", "A_dagger"))
def ppie(
    ys,
    probes,
    base_key=jax.random.PRNGKey(1776),
    gamma=0.75,
    epsilon=1e-8,
    beta_reg=0.1,
    N_iter=1000,
    obj_support=1,
    A=fft2_shifted,
    A_dagger=ifft2_shifted,
):
    x_est = obj_support * jnp.exp(
        1j * construct_random_phi(base_key, 0, probes[0].shape)
    )

    probe_int = jnp.abs(probes) ** 2
    denom = jnp.sum(probe_int, axis=0) + epsilon

    def update_step(_, x_est):
        phi_est = x_est * probes  # broadcasts over m (probe) axis
        y_est = A(phi_est)
        y_corr = jnp.abs(ys) * y_est / (jnp.abs(y_est) + epsilon)

        # See pie for explanation of beta parameter
        delta_phi = A_dagger(y_corr + (beta_reg - 1.0) * y_est) - beta_reg * phi_est

        x_est = (
            x_est + gamma * jnp.sum(jnp.conj(probes) * delta_phi, axis=0) / denom
        ) * obj_support

        return x_est

    x_est = lax.fori_loop(0, N_iter, update_step, x_est)
    return x_est


def ppie_multigpu(
    ys,
    probes,
    base_key=jax.random.PRNGKey(1776),
    gamma=0.75,
    epsilon=1e-8,
    N_iter=100,
    obj_support=jnp.ones(1),
    N_gpus=4,
):
    ppie_multigpu = jax.pmap(ppie, in_axes=(0, 0, 0, None, None, None, 0))

    keys = jax.random.split(base_key, N_gpus)

    ys_multigpu = jnp.stack([ys] * N_gpus)
    probes_multigpu = jnp.stack([probes] * N_gpus)
    # Object support in general can be an array so duplicate
    obj_support_multigpu = jnp.stack([obj_support] * N_gpus)

    x_est_multigpu = ppie_multigpu(
        ys_multigpu,
        probes_multigpu,
        keys,
        gamma,
        epsilon,
        N_iter,
        obj_support_multigpu,
    )
    return x_est_multigpu
