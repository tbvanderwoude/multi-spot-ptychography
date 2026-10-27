import secrets
from jax.tree_util import Partial
from jax import jit
from typing import Callable, Tuple, Any, Optional
import jax.numpy as jnp
from master.fft import *
import jax
from jax import lax

Solver = Callable[[jax.Array, int], Tuple[Any, bool, Any]]


@jit(static_argnames=("shape"))
def construct_random_phi(base_key, id, shape):
    key_i = jax.random.fold_in(base_key, id)
    phi = jax.random.uniform(key_i, shape, minval=-jnp.pi, maxval=jnp.pi)
    return phi


@jit
def fix_amplitude(amp, x):
    return amp * jnp.exp(1j * jnp.angle(x))


@jit
def limit_support(support, x):
    return support * x


@jit
def support_fixer(support):
    return Partial(limit_support, support)


@jit
def real_support(x):
    return jnp.real(x)


@jit
def real_pos_support(x):
    return jnp.maximum(jnp.real(x), 0.0)


@jit
def amplitude_fixer(amp):
    return Partial(fix_amplitude, amp)


@jit(static_argnames=("projector"))
def reflector_from_projector(projector, x):
    return 2 * projector(x) - x


@jit
def twin_image(x):
    return jnp.conj(jnp.flip(x, axis=(-2, -1)))


@jit(static_argnames=("projector_1", "projector_2"))
def RAAR_T(projector_1, projector_2, x):
    return 0.5 * (
        reflector_from_projector(projector_2, reflector_from_projector(projector_1, x))
        + x
    )


def gerchberg_saxton(amp_src, amp_tgt, **kwargs):
    return generic_ap_er_restart(
        fft2_shifted,
        ifft2_shifted,
        amplitude_fixer(amp_src),
        amplitude_fixer(amp_tgt),
        amp_src.shape,
        **kwargs,
    )


def error_reduction_positive(amp_tgt, F_pupil, **kwargs):
    return generic_ap_er_restart(
        fft2_shifted,
        ifft2_shifted,
        jax.jit(lambda x: jnp.maximum(jnp.real(support_fixer(F_pupil)(x)), 0)),
        amplitude_fixer(amp_tgt),
        amp_tgt.shape,
        **kwargs,
    )


def error_reduction_real(amp_tgt, F_pupil, **kwargs):
    return generic_ap_er_restart(
        fft2_shifted,
        ifft2_shifted,
        jax.jit(lambda x: jnp.real(support_fixer(F_pupil)(x))),
        amplitude_fixer(amp_tgt),
        amp_tgt.shape,
        **kwargs,
    )


def error_reduction_real_object(amp_tgt, F_pupil, **kwargs):
    return generic_ap_er_restart(
        fft2_shifted,
        ifft2_shifted,
        support_fixer(F_pupil),
        jax.jit(lambda x: jnp.real(amplitude_fixer(amp_tgt)(x))),
        amp_tgt.shape,
        **kwargs,
    )


def error_reduction(amp_tgt, F_pupil, **kwargs):
    return generic_ap_er_restart(
        fft2_shifted,
        ifft2_shifted,
        support_fixer(F_pupil),
        amplitude_fixer(amp_tgt),
        amp_tgt.shape,
        **kwargs,
    )


def hio(amp_tgt, obj_support, A=fft2_shifted, A_dagger=ifft2_shifted, **kwargs):
    return hybrid_input_output_restart(
        A,
        A_dagger,
        obj_support,
        lambda x: x,
        amplitude_fixer(amp_tgt),
        amp_tgt.shape,
        **kwargs,
    )


@jit
def run_no_restarts(
    run_fn: Solver,
    base_key: Optional[jax.Array] = None,
):
    if base_key is None:
        base_key = jax.random.PRNGKey(secrets.randbits(32))
    d, conv, err = run_fn(base_key, 0)
    return d, conv, err


def run_with_restarts(
    run_fn: Solver,
    restarts: int = 10,
    base_key: Optional[jax.Array] = None,
):
    if base_key is None:
        base_key = jax.random.PRNGKey(secrets.randbits(32))

    best_d = None
    best_err = jnp.inf
    last = None

    for r in range(restarts):
        d, conv, err = run_fn(base_key, r)
        last = (d, conv, err)

        if conv:
            return d, True, err

        err_val = err
        if best_d is None or err_val < best_err:
            best_d = d
            best_err = err_val

    return (best_d, False, best_err) if best_d is not None else last


def hybrid_input_output_no_restart(
    forward,
    adjoint,
    support_1,
    projection_1,
    projection_2,
    field_shape,
    base_key=None,
    **kwargs,
):
    run_fn = lambda base_key, r: generic_ap_hio(
        forward,
        adjoint,
        support_1,
        projection_1,
        projection_2,
        field_shape,
        base_key,
        r,
        **kwargs,
    )

    return run_no_restarts(run_fn, base_key=base_key)


def hybrid_input_output_restart(
    forward,
    adjoint,
    support_1,
    projection_1,
    projection_2,
    field_shape,
    restarts=10,
    base_key=None,
    **kwargs,
):
    run_fn = lambda base_key, r: generic_ap_hio(
        forward,
        adjoint,
        support_1,
        projection_1,
        projection_2,
        field_shape,
        base_key,
        r,
        **kwargs,
    )

    return run_with_restarts(run_fn, restarts=restarts, base_key=base_key)


def generic_ap_er_restart(
    forward,
    adjoint,
    projection_1,
    projection_2,
    field_shape,
    restarts=10,
    base_key=None,
    **kwargs,
):
    run_fn = lambda base_key, r: generic_ap_er(
        forward,
        adjoint,
        projection_1,
        projection_2,
        field_shape,
        base_key,
        r,
        **kwargs,
    )
    return run_with_restarts(run_fn, restarts=restarts, base_key=base_key)


@jit(
    static_argnames=(
        "forward",
        "adjoint",
        "projection_1",
        "projection_2",
        "field_shape",
    )
)
def generic_ap_hio(
    forward,
    adjoint,
    support_1,
    projection_1,
    projection_2,
    field_shape,
    base_key,
    id,
    beta=0.9,
    **kwargs,
):
    d_update = jax.jit(
        lambda c, d: projection_1((support_1 * c) + (1 - support_1) * (d - beta * c))
    )
    return generic_ap_plain(
        forward, adjoint, d_update, projection_2, field_shape, base_key, id, **kwargs
    )


def generic_ap_er(
    forward,
    adjoint,
    projection_1,
    projection_2,
    field_shape,
    base_key,
    id,
    **kwargs,
):
    return generic_ap_plain(
        forward,
        adjoint,
        jax.jit(lambda c, _: projection_1(c)),
        projection_2,
        field_shape,
        base_key,
        id,
        **kwargs,
    )


@jit(
    static_argnames=("forward", "adjoint", "f_d_update", "projection_2", "field_shape")
)
def generic_ap_plain(
    forward,
    adjoint,
    f_d_update,
    projection_2,
    field_shape,
    base_key,
    id,
    batch_dim=None,
    iterations=1000,
    initial_field=None,
    **kwargs,
):
    """Simplified AP scheme without stagnation checks to enable faster evaluation on GPU."""
    if batch_dim is not None:
        field_shape = (batch_dim,) + field_shape
    if initial_field is None:
        initial_phase = construct_random_phi(base_key, id, field_shape)
        initial_field = jnp.exp(1j * initial_phase)
    err_L2 = jnp.inf
    p_2 = jit(lambda x: adjoint(projection_2(forward(x))))

    def update_step(_, d):
        # c = p_2(d)
        return f_d_update(p_2(d), d)

    d = lax.fori_loop(0, iterations, update_step, initial_field)
    return d, True, err_L2


def generic_ap(
    forward,
    adjoint,
    f_d_update,
    projection_2,
    field_shape,
    base_key,
    id,
    batch_dim=None,
    iterations=1000,
    epsilon=1e-4,
    print_iter=100,
    stagnate_rel_delta=1e-6,
    initial_field=None,
):
    """Generic alternating projections (AP) scheme with a stagnation early-exit condition based on both
    - the difference between c and its projection d
    - the difference between d_k and d_{k+1} (fixpoint check)"""
    if batch_dim is not None:
        field_shape = (batch_dim,) + field_shape
    if initial_field is None:
        initial_phase = construct_random_phi(base_key, id, field_shape)
        initial_field = jnp.exp(1j * initial_phase)
    d = initial_field
    err_L2 = jnp.inf
    fixpoint_change = jnp.inf
    # Input - output picture: projection 2 as including the forward + adjoint steps
    p_2 = jit(lambda x: adjoint(projection_2(forward(x))))
    # min_err_L2 = jnp.inf
    # best_d = None
    for i in range(iterations):
        old_d = d
        c = p_2(d)
        d = f_d_update(c, d)
        old_err_L2 = err_L2
        old_fixpoint_change = fixpoint_change
        # Relative error between c and its projection onto set 1 in units of amplitude
        err_L2 = jnp.sqrt(jnp.sum(jnp.abs(d - c) ** 2) / jnp.sum(jnp.abs(c) ** 2))
        fixpoint_change = jnp.sqrt(
            jnp.sum(jnp.abs(d - old_d) ** 2) / jnp.sum(jnp.abs(old_d) ** 2)
        )
        # TODO: define/change batch convergence behavior
        # if err_L2 < min_err_L2:
        #     min_err_L2 = err_L2
        #     best_d = d
        # min_err_L2 = jnp.minimum(min_err_L2,err_L2)
        if err_L2 < epsilon:
            print(f"Final error (i = {i}): {err_L2}")
            return d, True, err_L2
        # if err_L2 >= min_err_L2 * 1.2 and i > iterations/2:
        #     print(f"Diverging (i = {i}): {err_L2}")
        #     if batch_dim is None:
        #         plot_intensity_phase(best_d)
        #     return best_d, False, min_err_L2
        rel_change = jnp.abs(old_err_L2 - err_L2) / old_err_L2
        rel_fixpoint_change = (
            jnp.abs(old_fixpoint_change - fixpoint_change) / old_fixpoint_change
        )
        if rel_change < stagnate_rel_delta and rel_fixpoint_change < stagnate_rel_delta:
            print(f"Stagnated (i = {i}): {err_L2}")
            return d, False, err_L2
        if i % print_iter == 0:
            print(f"i = {i}: error {err_L2}")
    return d, False, err_L2


def make_difference_map_step(p1, p2, beta):
    @jax.jit
    def step(d):
        p1d = p1(d)
        p2d = p2(d)

        f1 = (1.0 - 1.0 / beta) * p1d + (1.0 / beta) * d
        f2 = (1.0 + 1.0 / beta) * p2d - (1.0 / beta) * d

        return d + beta * (p1(f2) - p2(f1))

    return step


def make_difference_map_extract(p1, p2, beta):
    @jax.jit
    def extract(d):
        p1d = p1(d)
        p2d = p2(d)

        f1 = (1.0 - 1.0 / beta) * p1d + (1.0 / beta) * d
        f2 = (1.0 + 1.0 / beta) * p2d - (1.0 / beta) * d

        x1 = p1(f2)
        x2 = p2(f1)
        return x1, x2

    return extract


def generic_difference_map(
    forward,
    adjoint,
    projection_1,
    projection_2,
    field_shape,
    base_key,
    id,
    batch_dim=None,
    iterations=1000,
    epsilon=1e-4,
    print_iter=100,
    stagnate_rel_delta=1e-6,
    initial_field=None,
    beta=1.0,
):
    if batch_dim is not None:
        field_shape = (batch_dim,) + field_shape

    if initial_field is None:
        initial_phase = construct_random_phi(base_key, id, field_shape)
        initial_field = jnp.exp(1j * initial_phase)

    d = initial_field

    p1 = projection_1
    p2 = jax.jit(lambda x: adjoint(projection_2(forward(x))))

    dm_step = make_difference_map_step(p1, p2, beta)
    dm_extract = make_difference_map_extract(p1, p2, beta)

    err_L2 = jnp.inf
    fixpoint_change = jnp.inf

    for i in range(iterations):
        old_d = d
        d = dm_step(d)

        x1, x2 = dm_extract(d)

        old_err_L2 = err_L2
        old_fixpoint_change = fixpoint_change

        err_L2 = jnp.sqrt(jnp.sum(jnp.abs(x1 - x2) ** 2) / jnp.sum(jnp.abs(x1) ** 2))
        fixpoint_change = jnp.sqrt(
            jnp.sum(jnp.abs(d - old_d) ** 2) / jnp.sum(jnp.abs(old_d) ** 2)
        )

        if err_L2 < epsilon:
            print(f"Final error (i = {i}): {err_L2}")
            return x1, True, err_L2

        rel_change = jnp.abs(old_err_L2 - err_L2) / old_err_L2
        rel_fixpoint_change = (
            jnp.abs(old_fixpoint_change - fixpoint_change) / old_fixpoint_change
        )

        if rel_change < stagnate_rel_delta and rel_fixpoint_change < stagnate_rel_delta:
            print(f"Stagnated (i = {i}): {err_L2}")
            return x1, False, err_L2

        if i % print_iter == 0:
            print(f"i = {i}: error {err_L2}")

    x1, _ = dm_extract(d)
    return x1, False, err_L2


def difference_map_restart(
    forward,
    adjoint,
    projection_1,
    projection_2,
    field_shape,
    restarts=10,
    base_key=None,
    **kwargs,
):
    run_fn = lambda base_key, r: generic_difference_map(
        forward,
        adjoint,
        projection_1,
        projection_2,
        field_shape,
        base_key,
        r,
        **kwargs,
    )
    return run_with_restarts(run_fn, restarts=restarts, base_key=base_key)


def difference_map_no_restart(
    forward,
    adjoint,
    projection_1,
    projection_2,
    field_shape,
    base_key=None,
    **kwargs,
):
    run_fn = lambda base_key, r: generic_difference_map(
        forward,
        adjoint,
        projection_1,
        projection_2,
        field_shape,
        base_key,
        r,
        **kwargs,
    )
    return run_no_restarts(run_fn, base_key=base_key)


def difference_map(
    amp_tgt, obj_support, A=fft2_shifted, A_dagger=ifft2_shifted, **kwargs
):
    return difference_map_no_restart(
        A,
        A_dagger,
        support_fixer(obj_support),
        amplitude_fixer(amp_tgt),
        amp_tgt.shape,
        **kwargs,
    )
