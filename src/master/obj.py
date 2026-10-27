import jax.numpy as jnp
from skimage import io
from master.fft import fourier_axes, spatial_axes
from .imaging import make_circular_mask


def center_crop(img, N):
    H, W = img.shape[-2:]
    cy, cx = H // 2, W // 2
    hy = N // 2
    hx = N // 2
    return img[
        ...,
        cy - hy : cy + hy + (N % 2),
        cx - hx : cx + hx + (N % 2),
    ]


def normalize_image(raw_img):
    """Normalizes an image so that its dynamic range maps (in intensity) to 0-1"""
    min_img = jnp.min(raw_img)
    max_img = jnp.max(raw_img)
    normalized_img = (raw_img - min_img) / (max_img - min_img)
    return jnp.sqrt(normalized_img)


def normalize_image_intensity(x):
    """Normalizes an image so that its total intensity is 1"""
    return x / jnp.sqrt(jnp.sum(jnp.abs(x) ** 2, axis=(-2, -1)))


def pad_image(
    img,
    pad,  # int | (py, px) | ((top,bottom),(left,right))
    mode="constant",
    constant_value=0.0,
):
    if isinstance(pad, int):
        pad_width = ((pad, pad), (pad, pad))
    elif isinstance(pad, tuple) and len(pad) == 2 and isinstance(pad[0], int):
        py, px = pad
        pad_width = ((py, py), (px, px))
    else:
        pad_width = pad  # assume full ((t,b),(l,r))

    return jnp.pad(
        img,
        pad_width=pad_width,
        mode=mode,
        constant_values=constant_value if mode == "constant" else None,
    )


def image_to_norm_array(path, dir="images/"):
    raw_img = io.imread(dir + path).astype(jnp.float64)
    return normalize_image(raw_img)


def mandrill(dir="images/"):
    return image_to_norm_array("mandrill-256.png", dir)


def cameraman(dir="images/"):
    return image_to_norm_array("cameraman.png", dir)


def houseplants(dir="images/"):
    raw_img = io.imread(dir + "houseplants.png").astype(jnp.float64)
    return normalize_image(raw_img)


def laptop(dir="images/"):
    raw_img = io.imread(dir + "laptop.png").astype(jnp.float64)
    return normalize_image(raw_img)


def as_phase_field(x):
    return jnp.exp(2j * jnp.pi * x)


def crop_and_pad(x, N_crop=255, N_pad=256):
    cropped_x = center_crop(x, N_crop)

    obj_support_unpadded = jnp.ones(cropped_x.shape)

    x_obj = pad_image(cropped_x, N_pad)
    obj_support = pad_image(obj_support_unpadded, N_pad)
    return x_obj, obj_support


def create_physical_axes(x_obj, NA=0.5, lambda_0=655e-9, pixel_size=None, be=jnp):
    N = int(x_obj.shape[0])
    unit_l = 1.0 * lambda_0  # m

    k_0_SI = 2 * be.pi / lambda_0  # m-1
    k_0 = k_0_SI * unit_l

    # Higher: more oversampling, in theory easier to reconstruct but worse frequency support.
    size_factor = 4
    if pixel_size is None:
        pixel_size = lambda_0 / (size_factor * NA * unit_l)  # unit_l
    W = pixel_size * N
    y, x, Y, X, R = spatial_axes((N, N), dx=pixel_size, dy=pixel_size)
    f_y, f_x, F_Y, F_X = fourier_axes((N, N), dx=pixel_size, dy=pixel_size)

    k_x = 2 * be.pi * f_x
    k_y = 2 * be.pi * f_y
    K_X = 2 * be.pi * F_X
    K_Y = 2 * be.pi * F_Y
    k = be.sqrt(K_X**2 + K_Y**2)

    r_extent = [
        -W / 2,
        W / 2,
        -W / 2,
        W / 2,
    ]
    k_min = be.min(K_X)
    k_max = be.max(K_X)

    k_extent = [k_min, k_max, k_min, k_max]
    return x, y, X, Y, R, k_0, k_x, k_y, K_X, K_Y, k, r_extent, k_extent


def create_F_pupil(k, k_0, NA):
    F_pupil = make_circular_mask(k, k_0 * NA)
    nonzeros = jnp.sum(F_pupil, dtype=jnp.int32)
    samples = F_pupil.size
    spectral_oversampling = samples / nonzeros
    return F_pupil, spectral_oversampling


def analyse_oversampling(support):
    nonzeros = jnp.sum(jnp.abs(support) > 0.0, dtype=jnp.int32)
    samples = support.size
    oversampling = samples / nonzeros
    return oversampling
