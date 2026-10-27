from master.mle import norm_ll
import matplotlib.pyplot as plt
import scipy


def plot_abs_ll(lls):
    plt.plot(lls)
    plt.yscale("log")
    plt.show()


def plot_ll(lls, f_norm=norm_ll):
    plt.plot(f_norm(lls))
    plt.show()
    plot_abs_ll(lls)


def compare_lls(lls_1, lls_2, geometric_mean=False, labels=None, f_norm=norm_ll):
    if geometric_mean:
        if labels is not None:
            label_1 = labels[0]
            label_2 = labels[1]
        else:
            label_1 = None
            label_2 = None
        plt.plot(
            scipy.stats.gmean(1 - f_norm(lls_1), axis=1),
            label=label_1,
        )
        plt.plot(
            scipy.stats.gmean(1 - f_norm(lls_2), axis=1),
            linestyle="--",
            label=label_2,
        )
        if labels is not None:
            plt.legend()
    else:
        plt.plot(lls_1)
        plt.plot(lls_2, linestyle="--")
        plt.show()
        plt.plot(1 - f_norm(lls_1))
        plt.plot(1 - f_norm(lls_2), linestyle="--")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel(r"$N_{iter}$")
    plt.ylabel(r"$(\ell_{max} - \ell)/(\ell_{max} - \ell_{min})$")
    plt.show()
