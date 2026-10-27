import jax.numpy as jnp

lambda_0 = 655e-9  # m

unit_l = 1000.0 * lambda_0  # m

k_0_SI = 2 * jnp.pi / lambda_0  # m-1
k_0 = k_0_SI * unit_l

# Design NA
NA = 0.75
# Refractive index of cover slip
n_cov = 1.5142
p_SI = 9.504e-6  # m
d_cov_SI = 145e-6  # m
d_SI = 532e-6  # m

p = p_SI / unit_l
p_x, p_y = p, p
# Thickness of cover slip
d_cov = d_cov_SI / unit_l
# Free working distance
d = d_SI / unit_l

# Unit cell discretization
N = 256

# eps = 1e-24
eps = 1e-12
