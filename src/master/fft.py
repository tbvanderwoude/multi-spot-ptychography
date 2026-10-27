import jax.numpy as jnp
from jax import jit
from functools import partial


@jit
def fft_shifted(signal):
    return jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(signal), norm="ortho"))


@partial(jit, static_argnames=("axes",))
def fft2_shifted(signal, axes=(-2, -1)):
    return jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(signal, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


@partial(jit, static_argnames=("axes",))
def ifft2_shifted(signal, axes=(-2, -1)):
    return jnp.fft.fftshift(
        jnp.fft.ifft2(
            jnp.fft.ifftshift(signal, axes=axes),
            axes=axes,
            norm="ortho",
        ),
        axes=axes,
    )


@jit
def image_coordinates(im, W, H, pixel_size=1):
    x = jnp.linspace(-W / 2, W / 2, im.shape[0] + 1)[:-1] * pixel_size
    y = jnp.linspace(-H / 2, H / 2, im.shape[1] + 1)[:-1] * pixel_size
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return X, Y


@jit(static_argnames=("N"))
def create_axes(W, H, pixel_size=1, N=512):
    x = jnp.linspace(-W / 2, W / 2, N + 1)[:-1] * pixel_size
    y = jnp.linspace(-H / 2, H / 2, N + 1)[:-1] * pixel_size
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    R = jnp.sqrt(X**2 + Y**2)
    return X, Y, R


@jit
def ft_image_coordinates(im, W, H, pixel_size=1):
    x = jnp.linspace(-W / 2, W / 2, im.shape[0] + 1)[:-1] * pixel_size
    y = jnp.linspace(-H / 2, H / 2, im.shape[1] + 1)[:-1] * pixel_size
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    f_X, f_Y = to_ft_coordinates(X, Y)
    return f_X, f_Y


@jit
def to_ft_coordinates(X, Y):
    f_x = jnp.fft.fftfreq(X.shape[-1], d=X[..., 1] - X[..., 0])
    f_x = jnp.fft.fftshift(f_x)
    f_y = jnp.fft.fftfreq(Y.shape[-2], d=Y[..., 1, 0] - Y[..., 0, 0])
    f_y = jnp.fft.fftshift(f_y)
    f_X, f_Y = jnp.meshgrid(f_x, f_y, indexing="ij")
    return f_X, f_Y


@jit
def fft2c(x):
    # centered forward FFT
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(x), norm="ortho"))


@jit
def ifft2c(X):
    # centered inverse FFT
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(X), norm="ortho"))


@jit(static_argnames=("shape"))
def spatial_axes(shape, dx=1.0, dy=1.0):
    Nx, Ny = shape
    x = (jnp.arange(Nx) - (Nx // 2)) * dx
    y = (jnp.arange(Ny) - (Ny // 2)) * dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")  # matches array [i,j]
    R = jnp.sqrt(X**2 + Y**2)
    return x, y, X, Y, R


@jit(static_argnames=("shape"))
def fourier_axes(shape, dx=1.0, dy=1.0):
    Nx, Ny = shape
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(Nx, d=dx))
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(Ny, d=dy))
    FX, FY = jnp.meshgrid(fx, fy, indexing="ij")
    return fx, fy, FX, FY
