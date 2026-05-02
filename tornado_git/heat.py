import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def heatmap(
    v1, v2,
    bins=40,
    sigma=1,
    xlim=None,
    ylim=None,
    figsize=(6, 5),
    vmin=None,
    vmax=None
):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError("v1 and v2 must both be 1D arrays.")
    if len(v1) != len(v2):
        raise ValueError("v1 and v2 must have the same length.")

    hist_range = None
    if xlim is not None and ylim is not None:
        hist_range = [xlim, ylim]
    elif xlim is not None or ylim is not None:
        raise ValueError("Please provide both xlim and ylim, or neither.")

    # histogram of counts
    H, xedges, yedges = np.histogram2d(v1, v2, bins=bins, range=hist_range)

    # smooth counts
    H = gaussian_filter(H, sigma=sigma)

    # convert to density
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    H = H / (H.sum() * dx * dy)

    plt.figure(figsize=figsize)
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(label="Density")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title("Density heat map of v1 and v2")

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.show()
    
    
def four_heatmaps(
    X_test, X_TT, X_vae, X_marf,
    dim_1, dim_2,
    bins=100,
    sigma=1,
    xlim=(-0.2, 1.2),
    ylim=(-0.2, 1.2),
    vmin=None,
    vmax=None,
    density=True,
    figsize=(16, 4),
    title_fontsize=16,
    label_fontsize=14
):
    data_list = [
        (X_test[:, dim_1], X_test[:, dim_2], "Test Data"),
        (X_TT[:, dim_1],   X_TT[:, dim_2],   "TT"),
        (X_vae[:, dim_1],  X_vae[:, dim_2],  "Variational Autoencoder"),
        (X_marf[:, dim_1], X_marf[:, dim_2], "Diffusion"),
    ]

    hist_range = [xlim, ylim]
    H_list = []

    for v1, v2, _ in data_list:
        H, xedges, yedges = np.histogram2d(v1, v2, bins=bins, range=hist_range)
        H = gaussian_filter(H, sigma=sigma)

        if density:
            dx = xedges[1] - xedges[0]
            dy = yedges[1] - yedges[0]
            H = H / (H.sum() * dx * dy)

        H_list.append(H)

    if vmin is None:
        vmin = min(H.min() for H in H_list)
    if vmax is None:
        vmax = max(H.max() for H in H_list)

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)

    for ax, H, (_, _, title) in zip(axes, H_list, data_list):
        ax.imshow(
            H.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(f"dim {dim_1}", fontsize=label_fontsize)
        ax.set_ylabel(f"dim {dim_2}", fontsize=label_fontsize)

        # remove numeric tick labels on both axes
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # return H_list, xedges, yedges