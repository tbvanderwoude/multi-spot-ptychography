from master import *
from master.operators import make_pty_ops_and_support
from master.probes import (
    build_spectral_support_rel,
    calc_array_properties,
    construct_probe_fields,
    construct_probe_inf_probe_approx,
    create_probe_Rm,
)
from dataclasses import dataclass
from dacite import from_dict

from master.spotgen import create_multispot_array_product

# Common definitions used in thesis ptychography simulations to avoid mismatches


def cameraman_object(N_crop, N_pad):
    x_cameraman = cameraman() * 0.9 + 0.1
    # TODO: make estimator statistics more robust to avoid needing to stay away from the 'cut'
    x_mandrill = as_phase_field(mandrill() ** 2 - 0.5)
    x_obj, obj_support = crop_and_pad(
        x_cameraman * x_mandrill, N_crop=N_crop, N_pad=N_pad
    )
    return x_obj, obj_support


def pupil_crop(F_pupil):
    tol = 0.01
    r, c = np.nonzero(np.abs(np.array(F_pupil)) >= tol)
    r0, r1 = (
        r.min(),
        r.max() + 1,
    )
    c0, c1 = c.min(), c.max() + 1
    print(r0, r1, c0, c1)
    N_pupil_crop = r1 - r0
    return N_pupil_crop


def create_ops_and_pupil(N_crop, N_pad, k, k_0, NA_det, fast_ops=True):
    N_full = N_crop + N_pad * 2
    F_pupil = make_circular_mask(k, k_0 * NA_det)

    N_pupil_crop = pupil_crop(F_pupil)

    if fast_ops:
        A, A_dagger, pupil_support = make_pty_ops_and_support(
            N_full, N_pupil_crop, F_pupil
        )
    else:
        A = jit(lambda x: F_pupil * fft2_shifted(x))
        A_dagger = jit(lambda x: ifft2_shifted(F_pupil * x))
    return A, A_dagger, F_pupil, pupil_support


def create_probe_with_average_intensity(
    N_crop,
    x_obj,
    A,
    probe_type,
    probe_seed,
    NA_spot,
    L_x,
    L_y,
    frac_x,
    frac_y,
    r_extent,
    M,
    K_X,
    K_Y,
    k,
    k_0,
    cdtype,
    N_int_avg,
    f_probe=create_multispot_array_product,
    delta_r_coeff=1.0,
    use_shift=True,
    **kwargs,
):
    probes, _, _ = construct_probe_fields(
        probe_type,
        probe_seed,
        NA_spot,
        L_x,
        L_y,
        frac_x,
        frac_y,
        r_extent,
        M,
        1.0,
        K_X,
        K_Y,
        k,
        k_0,
        cdtype,
        f_probe=f_probe,
        delta_r_coeff=delta_r_coeff,
        use_shift=use_shift,
        **kwargs,
    )
    det_sum_unscaled = jnp.sum(jnp.abs(A(probes * x_obj)) ** 2)
    N_photons = N_crop**2 * N_int_avg
    N_coeff = float(N_photons / (det_sum_unscaled))
    probes *= np.sqrt(N_coeff)
    return probes, N_photons, N_coeff


@dataclass
class ProbeParams:
    probe_type: str
    probe_seed: int
    NA_spot: float
    L_x: float
    L_y: float
    frac_x: float
    frac_y: float

    def from_dict(probe_dict):
        return from_dict(data_class=ProbeParams, data=probe_dict)


class PtySim:
    def __init__(
        self,
        N_crop,
        N_pad,
        NA_det,
        lambda_0=1.0,
        unit_l=1.0,
        object="cameraman",
        x_true=None,
        obj_support=None,
        be=jnp,
    ):
        self.N_crop = N_crop
        self.N_pad = N_pad
        self.NA_det = NA_det
        self.lambda_0 = lambda_0
        self.unit_l = unit_l
        if object == "cameraman":
            x_true, obj_support = cameraman_object(N_crop, N_pad)
        self.x_true = x_true
        self.F_pupil = None
        self.obj_support = obj_support
        self.cdtype = jnp.array(1j).dtype
        self.A = None
        self.be = be

    # TODO: start using this to avoid useless materialization of all probe fields AND all probe object fields
    def construct_probe_implicit(
        probe_type,
        probe_seed,
        NA_spot,
        L_x,
        L_y,
        frac_x,
        frac_y,
        r_extent,
        M,
        K_X,
        K_Y,
        k,
        k_0,
        f_probe=create_multispot_array_product,
        delta_r_coeff=1.0,
    ):
        array_bounds, array_period_x, array_period_y = calc_array_properties(
            L_x, L_y, frac_x, frac_y, r_extent
        )
        print(
            f"array_period_x={array_period_x:.3g}, array_period_y={array_period_y:.3g}"
        )
        # Define 25 probe positions using random distribution in +-3 p_x central area
        r = array_period_x
        delta_r = r * delta_r_coeff

        R_probes = create_probe_Rm(
            M,
            delta_r,
            probe_seed,
            probe_type,
        )
        return R_probes

    def get_R_probes(self, M, delta_r, pp: ProbeParams):
        return create_probe_Rm(
            M,
            delta_r,
            pp.probe_seed,
            pp.probe_type,
        )

    def create_axes(self):
        return create_physical_axes(
            self.x_true,
            lambda_0=self.lambda_0,
            pixel_size=self.lambda_0 / (2 * self.unit_l),
            be=self.be,
        )

    def get_ops_and_pupil(self):
        if self.A == None:
            x, y, X, Y, R, k_0, k_x, k_y, K_X, K_Y, k, r_extent, k_extent = (
                self.create_axes()
            )
            A, A_dagger, F_pupil, pupil_support = create_ops_and_pupil(
                self.N_crop, self.N_pad, k, k_0, self.NA_det
            )
            self.A = A
            self.A_dagger = A_dagger
            self.F_pupil = np.asarray(F_pupil)
            self.pupil_support = np.asarray(pupil_support)
        return self.A, self.A_dagger, self.F_pupil, self.pupil_support

    def create_probe_with_average_intensity(
        self,
        M,
        pp: ProbeParams,
        pixel_intensity,
        f_probe=create_multispot_array_product,
        **kwargs,
    ):
        return self.create_probe_with_average_intensity_expanded(
            M,
            pp.probe_type,
            pp.probe_seed,
            pp.NA_spot,
            pp.L_x,
            pp.L_y,
            pp.frac_x,
            pp.frac_y,
            pixel_intensity,
            f_probe,
            **kwargs,
        )

    def create_base_probe(
        self,
        pp: ProbeParams,
        pixel_intensity,
        f_probe=create_multispot_array_product,
        **kwargs,
    ):
        return self.create_probe_with_average_intensity_expanded(
            1,
            pp.probe_type,
            pp.probe_seed,
            pp.NA_spot,
            pp.L_x,
            pp.L_y,
            pp.frac_x,
            pp.frac_y,
            pixel_intensity,
            f_probe,
            delta_r_coeff=0.0,
            **kwargs,
        )

    def create_probe_with_average_intensity_expanded(
        self,
        M,
        probe_type,
        probe_seed,
        NA_spot,
        L_x,
        L_y,
        frac_x,
        frac_y,
        pixel_intensity,
        f_probe=create_multispot_array_product,
        **kwargs,
    ):
        A, A_dagger, F_pupil, pupil_support = self.get_ops_and_pupil()
        x, y, X, Y, R, k_0, k_x, k_y, K_X, K_Y, k, r_extent, k_extent = (
            self.create_axes()
        )
        probes, N_photons, N_coeff = create_probe_with_average_intensity(
            self.N_crop,
            self.x_true,
            A,
            probe_type,
            probe_seed,
            NA_spot,
            L_x,
            L_y,
            frac_x,
            frac_y,
            r_extent,
            M,
            K_X,
            K_Y,
            k,
            k_0,
            self.cdtype,
            pixel_intensity,
            f_probe,
            **kwargs,
        )
        return probes, N_photons, N_coeff

    def contruct_spectral_support(self, base_probe):
        self.get_ops_and_pupil()
        p_inf_spect, N_nonzero_orders, spectral_support = (
            construct_probe_inf_probe_approx(base_probe, self.F_pupil)
        )
        P_spectral_support = lambda x: ifft2_shifted(fft2_shifted(x) * spectral_support)
        return p_inf_spect, N_nonzero_orders, spectral_support, P_spectral_support

    def construct_spectral_support_rel(
        self, base_probe, rel_threshold=1e-5, cross_correlation=False
    ):
        self.get_ops_and_pupil()
        spectrum_filtered_probe, spectral_support = build_spectral_support_rel(
            base_probe,
            self.F_pupil,
            rel_threshold=rel_threshold,
            cross_correlation=cross_correlation,
        )
        P_spectral_support = lambda x: ifft2_shifted(fft2_shifted(x) * spectral_support)
        return spectrum_filtered_probe, spectral_support, P_spectral_support
