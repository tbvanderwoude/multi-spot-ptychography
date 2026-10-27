import jax.numpy as jnp
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib as mpl

mpl.rcParams["savefig.bbox"] = "standard"  # valid values are 'tight'/'standard'
mpl.rcParams["savefig.pad_inches"] = 0.0

PLOT_DPI = 300
CBAR_FRACTION = 0.045
CBAR_PAD = 0.02
SPACE_X_LABEL = r"$x'$ (px)"
SPACE_Y_LABEL = r"$y'$ (px)"
SPACE_X_LAMBDA_LABEL = r"$x'$ ($\lambda$)"
SPACE_Y_LAMBDA_LABEL = r"$y'$ ($\lambda$)"
FREQ_X_LABEL = r"$k_{x'}$ (rad/px)"
FREQ_Y_LABEL = r"$k_{y'}$ (rad/px)"
FREQ_X_LAMBDA_LABEL = r"$k_{x'}$ (rad/$\lambda$)"
FREQ_Y_LAMBDA_LABEL = r"$k_{y'}$ (rad/$\lambda$)"
X_LABEL_PAD = 2.0
Y_LABEL_PAD = -4.0

CBAR_ARGS = {"fraction": CBAR_FRACTION, "pad": CBAR_PAD}
PI_TICKS = [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi]
PI_TICK_LABELS = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
TWO_PI_TICKS = [-2 * jnp.pi, -jnp.pi, 0, jnp.pi, jnp.pi * 2.0]
TWO_PI_TICK_LABELS = [r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"]

FIG_W = 6
FIG_H = 4.75


def plot_lambda_spect(
    y,
    extent=None,
    use_log=False,
    label="",
    fontsize=12,
    save_path=None,
    dpi=PLOT_DPI,
    x_label=FREQ_X_LAMBDA_LABEL,
    y_label=FREQ_Y_LAMBDA_LABEL,
    use_two_pi_extent=True,
    use_rad_extent=False,
    patch=None,
):
    return plot(
        y,
        extent,
        use_log,
        label,
        fontsize,
        save_path,
        dpi,
        x_label,
        y_label,
        use_two_pi_extent,
        use_rad_extent,
        patch,
    )


def plot_lambda(
    y,
    extent=None,
    use_log=False,
    label="",
    fontsize=12,
    save_path=None,
    dpi=PLOT_DPI,
    x_label=SPACE_X_LAMBDA_LABEL,
    y_label=SPACE_Y_LAMBDA_LABEL,
    use_two_pi_extent=False,
    use_rad_extent=False,
    patch=None,
):
    return plot(
        y,
        extent,
        use_log,
        label,
        fontsize,
        save_path,
        dpi,
        x_label,
        y_label,
        use_two_pi_extent,
        use_rad_extent,
        patch,
    )


def plot(
    y,
    extent=None,
    use_log=False,
    label="",
    fontsize=12,
    save_path=None,
    dpi=PLOT_DPI,
    x_label=SPACE_X_LABEL,
    y_label=SPACE_Y_LABEL,
    use_two_pi_extent=False,
    use_rad_extent=False,
    patch=None,
):
    norm = LogNorm() if use_log else None
    fig = plt.figure(figsize=(FIG_W, FIG_H), layout="compressed")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0)
    im = plt.imshow(y, extent=extent, norm=norm)
    cbar = plt.colorbar(im, **CBAR_ARGS)
    cbar.set_label(label, fontsize=fontsize)
    plt.xlabel(x_label, labelpad=X_LABEL_PAD)
    plt.ylabel(y_label, labelpad=Y_LABEL_PAD)
    if use_rad_extent:
        plt.xticks(PI_TICKS, PI_TICK_LABELS)
        plt.yticks(PI_TICKS, PI_TICK_LABELS)
    if use_two_pi_extent:
        plt.xticks(TWO_PI_TICKS, TWO_PI_TICK_LABELS)
        plt.yticks(TWO_PI_TICKS, TWO_PI_TICK_LABELS)
    if patch is not None:
        plt.gca().add_patch(patch)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
    plt.show()


def plot_row(ys, extent=None, labels=None, vertical=False):
    """Plots a row of or column (for vertical=True) of images with optional labels. Labels are already in a LaTeX math environment."""
    N = len(ys)
    if vertical:
        fig, ax = plt.subplots(
            N, 1, figsize=(FIG_W, FIG_H * N), constrained_layout=True
        )
    else:
        fig, ax = plt.subplots(
            1, N, figsize=(FIG_W * N, FIG_H), constrained_layout=True
        )

    for i, y in enumerate(ys):
        im = ax[i].imshow(y, extent=extent)
        fig.colorbar(im, ax=ax[i], **CBAR_ARGS)
        if labels is not None:
            ax[i].set_title(rf"${labels[i]}$")
    plt.show()


def plot_intensity_phase_spect(
    y,
    extent=None,
    circle=None,
    label="x",
    use_log=True,
    vertical=False,
    save_path=None,
    dpi=PLOT_DPI,
    intensity_unit="a.u.",
    x_label=FREQ_X_LABEL,
    y_label=FREQ_Y_LABEL,
    use_rad_extent=False,
    use_two_pi_extent=False,
):
    return plot_intensity_phase(
        y,
        extent,
        circle,
        label,
        use_log,
        vertical,
        save_path,
        dpi,
        intensity_unit,
        x_label,
        y_label,
        use_rad_extent,
        use_two_pi_extent,
    )


def plot_intensity_phase_spect_lambda(
    y,
    extent=None,
    circle=None,
    label="x",
    use_log=True,
    vertical=False,
    save_path=None,
    dpi=PLOT_DPI,
    intensity_unit="a.u.",
    x_label=FREQ_X_LAMBDA_LABEL,
    y_label=FREQ_Y_LAMBDA_LABEL,
    use_rad_extent=False,
    use_two_pi_extent=True,
):
    return plot_intensity_phase(
        y,
        extent,
        circle,
        label,
        use_log,
        vertical,
        save_path,
        dpi,
        intensity_unit,
        x_label,
        y_label,
        use_rad_extent,
        use_two_pi_extent,
    )


def plot_intensity_phase_lambda(
    y,
    extent=None,
    circle=None,
    label="x",
    use_log=False,
    vertical=False,
    save_path=None,
    dpi=PLOT_DPI,
    intensity_unit="a.u.",
    x_label=SPACE_X_LAMBDA_LABEL,
    y_label=SPACE_Y_LAMBDA_LABEL,
    use_rad_extent=False,
    use_two_pi_extent=False,
):
    return plot_intensity_phase(
        y,
        extent,
        circle,
        label,
        use_log,
        vertical,
        save_path,
        dpi,
        intensity_unit,
        x_label,
        y_label,
        use_rad_extent,
        use_two_pi_extent,
    )


def plot_intensity_phase(
    y,
    extent=None,
    circle=None,
    label="x",
    use_log=False,
    vertical=False,
    save_path=None,
    dpi=PLOT_DPI,
    intensity_unit="a.u.",
    x_label=SPACE_X_LABEL,
    y_label=SPACE_Y_LABEL,
    use_rad_extent=False,
    use_two_pi_extent=False,
):
    norm = LogNorm() if use_log else None
    I = jnp.abs(y) ** 2
    phi = jnp.angle(jnp.where(y == 0, 0.0, y))
    if vertical:
        fig, ax = plt.subplots(2, 1, figsize=(FIG_W, FIG_H * 2), layout="compressed")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H), layout="compressed")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0, hspace=0)
    im0 = ax[0].imshow(I, extent=extent, norm=norm)

    cbar0 = fig.colorbar(im0, ax=ax[0], norm=norm, **CBAR_ARGS)
    cbar0.set_label(rf"$|{label}|^2$ ({intensity_unit})")
    im1 = ax[1].imshow(phi, extent=extent, vmin=-jnp.pi, vmax=jnp.pi)
    cbar1 = fig.colorbar(im1, ax=ax[1], **CBAR_ARGS)
    cbar1.set_label(rf"$\angle {label}$ (rad)")
    cbar1.ax.yaxis.set_major_locator(FixedLocator(PI_TICKS))
    cbar1.ax.yaxis.set_major_formatter(FixedFormatter(PI_TICK_LABELS))
    for i in range(2):
        ax[i].set_xlabel(x_label, labelpad=X_LABEL_PAD)  # or whatever units
        ax[i].set_ylabel(y_label, labelpad=Y_LABEL_PAD)
        if use_rad_extent:
            ax[i].set_xticks(PI_TICKS, PI_TICK_LABELS)
            ax[i].set_yticks(PI_TICKS, PI_TICK_LABELS)
        elif use_two_pi_extent:
            ax[i].set_xticks(TWO_PI_TICKS, TWO_PI_TICK_LABELS)
            ax[i].set_yticks(TWO_PI_TICKS, TWO_PI_TICK_LABELS)
    if circle is not None:
        ax[0].add_patch(circle)
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi)
    plt.show()
