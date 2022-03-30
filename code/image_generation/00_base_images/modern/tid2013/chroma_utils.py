import numpy as np
import matplotlib.pyplot as plt

def plot_lab_image_structure(lab_image: np.array):
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(ncols=2, nrows=2, figure=fig)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_chroma = fig.add_subplot(gs[1, 1])

    # L value plot.
    max_l = np.round(np.amax(lab_image[:, :, 0]), 1)
    min_l = np.round(np.amin(lab_image[:, :, 0]), 1)
    mesh = ax_l.pcolormesh(np.flip(lab_image[:, :, 0], axis=0), cmap='gray')
    fig.colorbar(mesh, ax=ax_l)
    ax_l.set_title(f'L* values: {min_l} <= l <= {max_l}')

    # a value plot.
    max_a = np.round(np.amax(lab_image[:, :, 1]), 1)
    min_a = np.round(np.amin(lab_image[:, :, 1]), 1)
    mesh = ax_a.pcolormesh(np.flip(lab_image[:, :, 1], axis=0), cmap='RdYlGn_r')
    fig.colorbar(mesh, ax=ax_a)
    ax_a.set_title(f'a* values: {min_a} <= a <= {max_a}')

    # b value plot.
    max_b = np.round(np.amax(lab_image[:, :, 2]), 1)
    min_b = np.round(np.amin(lab_image[:, :, 2]), 1)
    mesh = ax_b.pcolormesh(np.flip(lab_image[:, :, 2], axis=0))
    fig.colorbar(mesh, ax=ax_b)
    ax_b.set_title(f'b* values: {min_b} <= b <= {max_b}')

    # Chroma plot.
    chroma_matrix = np.sqrt((lab_image[:, :, 1] * lab_image[:, :, 1]) + (lab_image[:, :, 2] * lab_image[:, :, 2]))
    max_chroma = np.round(np.amax(chroma_matrix), 1)
    min_chroma = np.round(np.amin(chroma_matrix), 1)
    mesh = ax_chroma.pcolormesh(np.flip(chroma_matrix, axis=0), cmap='plasma')
    #axs_chroma.set_colorbar(mesh)
    ax_chroma.set_title(f'Chroma: {min_chroma} <= C <= {max_chroma}')

    fig.colorbar(mesh, ax=ax_chroma)
    plt.show()