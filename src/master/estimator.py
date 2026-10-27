from jax import jit
from .stats import *
import jax.numpy as jnp
from .plotting import plot


def evaluate_estimator(x_model_ml, x_obj, obj_support, label="HIO+ML", extent=None):
    n_hat = np.abs(x_model_ml) ** 2 - jnp.var(x_model_ml, axis=0, ddof=1, keepdims=True)
    n = jnp.abs(x_obj) ** 2
    n_hat_mu, n_hat_std, n_hat_bias, n_hat_MSE = estimator_stats(n_hat, n)
    print("n_k")
    plot(n_hat_mu, label=r"$[\mu_{\hat{n}_k}]^{" + label + r"}$", extent=extent)
    plot(n, label=r"$n_k$", extent=extent)
    plot(n_hat_bias, label=r"$[\mu_{\hat{n}_k}]^{" + label + r"} - n_k$", extent=extent)
    plot(n_hat_std, label=r"$[\sigma_{\hat{n}_k}]^{" + label + r"}$", extent=extent)
    plot(n_hat_MSE, label=r"$[MSE_{\hat{n}_k}]^{" + label + r"}$", extent=extent)

    print("phi_k")

    phi_hat = np.angle(x_model_ml) * obj_support
    phi = jnp.angle(x_obj) * obj_support

    # Circular mean of the phase estimates
    phi_hat_mu = np.angle(np.mean(np.exp(1j * phi_hat), axis=0))

    # Wrapped phase errors in (-pi, pi]
    phi_hat_error = np.angle(np.exp(1j * (phi_hat - phi)))

    # Statistics in the local phase coordinate
    phi_hat_bias = np.mean(phi_hat_error, axis=0)
    phi_hat_std = np.std(phi_hat_error, axis=0, ddof=1)
    phi_hat_MSE = np.mean(phi_hat_error**2, axis=0)

    plot(phi_hat_mu, label=r"$[\mu_{\hat{\phi}_k}]^{" + label + r"}$", extent=extent)
    plot(phi, label=r"$\phi_k$", extent=extent)
    plot(
        phi_hat_bias,
        label=r"$[\mu_{\hat{\phi}_k}]^{" + label + r"} - \phi_k$",
        extent=extent,
    )
    plot(
        phi_hat_std, label=r"$[\sigma_{\hat{\phi}_k}]^{" + label + r"}$", extent=extent
    )
    plot(phi_hat_MSE, label=r"$[MSE_{\hat{\phi}_k}]^{" + label + r"}$", extent=extent)
    return n_hat_mu, n_hat_std, phi_hat_mu, phi_hat_std


def evaluate_estimator_uv(
    x_model_ml, x_obj, obj_support, factor, label="HIO+ML", extent=None
):
    u_hat = np.real(x_model_ml)
    v_hat = np.imag(x_model_ml)
    u = np.real(x_obj)
    v = np.imag(x_obj)
    # Re x / u block.

    u_hat_mu, u_hat_std, u_hat_bias, u_hat_MSE = estimator_stats(u_hat, u)

    # Im x / v block.
    v_hat_mu, v_hat_std, v_hat_bias, v_hat_MSE = estimator_stats(v_hat, v)
    print("u_k")
    plot(
        u_hat_mu * factor,
        label=r"$\operatorname{Mean}(\hat{u}_k)$",
        extent=extent,
    )
    plot(
        u_hat_bias * factor,
        label=r"$\operatorname{Bias}(\hat{u}_k)$",
        extent=extent,
    )
    plot(
        u_hat_std * factor,
        label=r"$\operatorname{Std}(\hat{u}_k)$",
        extent=extent,
    )
    plot(
        u_hat_MSE * factor**2,
        label=r"$\operatorname{MSE}(\hat{u}_k)$",
        extent=extent,
    )

    print("v_k")
    plot(
        v_hat_mu * factor,
        label=r"$\operatorname{Mean}(\hat{v}_k)$",
        extent=extent,
    )
    plot(
        v_hat_bias * factor,
        label=r"$\operatorname{Bias}(\hat{v}_k)$",
        extent=extent,
    )
    plot(
        v_hat_std * factor,
        label=r"$\operatorname{Std}(\hat{v}_k)$",
        extent=extent,
    )
    plot(
        v_hat_MSE * factor**2,
        label=r"$\operatorname{MSE}(\hat{v}_k)$",
        extent=extent,
    )

    return (
        u_hat_mu,
        u_hat_std,
        u_hat_bias,
        u_hat_MSE,
        v_hat_mu,
        v_hat_std,
        v_hat_bias,
        v_hat_MSE,
    )


@jit
def reconstruction_error_abs(x, x_true):
    return jnp.linalg.norm(x - x_true, axis=(-2, -1))


@jit
def reconstruction_error_rel(x, x_true):
    return reconstruction_error_abs(x, x_true) / jnp.linalg.norm(x_true, axis=(-2, -1))


@jit
def reconstruction_bias_abs(x, x_true):
    return jnp.linalg.norm(jnp.mean(x, axis=0) - x_true, axis=(-2, -1))


@jit
def reconstruction_std_abs(x):
    return jnp.sqrt(
        jnp.mean(
            jnp.linalg.norm(x - jnp.mean(x, axis=0, keepdims=True), axis=(-2, -1)) ** 2,
            axis=0,
        )
    )


@jit
def reconstruction_rmse_abs(x, x_true):
    return jnp.sqrt(jnp.mean(reconstruction_error_abs(x, x_true) ** 2, axis=0))


@jit
def reconstruction_bias_rel(x, x_true):
    return reconstruction_bias_abs(x, x_true) / jnp.linalg.norm(x_true, axis=(-2, -1))


@jit
def reconstruction_std_rel(x, x_true):
    return reconstruction_std_abs(x) / jnp.linalg.norm(x_true, axis=(-2, -1))


@jit
def reconstruction_rmse_rel(x, x_true):
    return jnp.sqrt(jnp.mean(reconstruction_error_rel(x, x_true) ** 2, axis=0))


@jit
def reconstruction_bias_abs_unbiased(x, x_true):
    N_samples = x.shape[0]
    return jnp.sqrt(
        jnp.maximum(
            reconstruction_bias_abs(x, x_true) ** 2
            - reconstruction_std_abs(x) ** 2 / N_samples,
            0.0,
        )
    )


@jit
def reconstruction_bias_rel_unbiased(x, x_true):
    return reconstruction_bias_abs_unbiased(x, x_true) / jnp.linalg.norm(
        x_true, axis=(-2, -1)
    )
