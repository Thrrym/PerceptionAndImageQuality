import numpy as np
import matplotlib.pyplot as plt

def save_plot_lab_image_structure(lab_image: np.array, title: str, path):
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(ncols=2, nrows=2, figure=fig)

    ax_l = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_chroma = fig.add_subplot(gs[1, 1])

    # L value plot.
    max_l = np.around(np.amax(lab_image[:, :, 0]))
    min_l = np.around(np.amin(lab_image[:, :, 0]))
    mesh = ax_l.pcolormesh(np.flip(lab_image[:, :, 0], axis=0), cmap='gray')
    fig.colorbar(mesh, ax=ax_l)
    ax_l.set_title(f'L* values: {min_l} <= l <= {max_l}')

    # a value plot.
    max_a = np.around(np.amax(lab_image[:, :, 1]))
    min_a = np.around(np.amin(lab_image[:, :, 1]))
    mesh = ax_a.pcolormesh(np.flip(lab_image[:, :, 1], axis=0), cmap='RdYlGn_r')
    fig.colorbar(mesh, ax=ax_a)
    ax_a.set_title(f'a* values: {min_a} <= a <= {max_a}')

    # b value plot.
    max_b = np.around(np.amax(lab_image[:, :, 2]))
    min_b = np.around(np.amin(lab_image[:, :, 2]))
    mesh = ax_b.pcolormesh(np.flip(lab_image[:, :, 2], axis=0))
    fig.colorbar(mesh, ax=ax_b)
    ax_b.set_title(f'b* values: {min_b} <= b <= {max_b}')

    # Chroma plot.
    chroma_matrix = np.sqrt((lab_image[:, :, 1] * lab_image[:, :, 1]) + (lab_image[:, :, 2] * lab_image[:, :, 2]))
    max_chroma = np.around(np.amax(chroma_matrix))
    min_chroma = np.around(np.amin(chroma_matrix))
    mesh = ax_chroma.pcolormesh(np.flip(chroma_matrix, axis=0), cmap='plasma')
    #axs_chroma.set_colorbar(mesh)
    ax_chroma.set_title(f'Chroma: {min_chroma} <= C <= {max_chroma}')

    fig.suptitle(title)
    fig.colorbar(mesh, ax=ax_chroma)
    
    plt.savefig(path, facecolor='white')
    plt.close('all')