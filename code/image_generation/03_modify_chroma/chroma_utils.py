import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from scipy.optimize import curve_fit
from lab_exceptions import *

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

def get_max_chroma_factor(image_to_modify):
    min_a = np.abs(np.amin(image_to_modify[:, :, 1]))
    max_a = np.abs(np.amax(image_to_modify[:, :, 1]))
    min_b = np.abs(np.amin(image_to_modify[:, :, 2]))
    max_b = np.abs(np.amax(image_to_modify[:, :, 2]))
    max_abs_value = max(min_a, max_a, min_b, max_b)
    ab_abs_max = 99
    return ab_abs_max / max_abs_value

def get_factors_between_min_max(min_value, max_value, plot_path=None) -> list:
    

    def fit_func(x, a, b):
        return a*x + b

    popt, pcov = curve_fit(fit_func, [min_value, max_value], [0, 8])
    print(popt)

    factor_num_1 = int(fit_func(1.0, *popt))
    if factor_num_1 < 1.0:
        factor_num_1 = 1.0
    fixed_factors_y = [min_value, 1.0, max_value]
    fixed_factors_x = [0, factor_num_1, 8]
    print(fixed_factors_x)
    #calculated_function = CubicSpline(fixed_factors_x, fixed_factors_y)
    calculated_function = Akima1DInterpolator(fixed_factors_x, fixed_factors_y)

    x_range = np.arange(9)
    #plt.scatter(x_range, calculated_function(x_range))
    #plt.show()

    return list(calculated_function(x_range))

def get_modification_matrix(chroma_factor, requested_shape):
    vector_to_multiply_elementwise = np.array([1, chroma_factor, chroma_factor])
    return np.tile(vector_to_multiply_elementwise[None][None], (requested_shape[0], requested_shape[1], 1))

def modify_lab_image_chroma(chroma_factor, image_to_modify):
    """ Modify the provided image in Lab color space by given chroma factor. """
    # Generate the modifier matrix.
    modifier = get_modification_matrix(chroma_factor, image_to_modify.shape)

    # Elementwise multiplication of original image.
    modified_image = image_to_modify * modifier
    #return modified_image
    # Ensure the Lab color space bounds are not violated.
    # 0 <= L <= 100
    L_max = 100
    L_min = -100
    ab_max = 100
    ab_min = -100
    if np.less(np.amin(modified_image[:, :, 0]), L_min) or np.greater(np.amax(modified_image[:, :, 0]), L_max):
        print('L* value out of bound.')
        raise LabValueOutOfBound
    # -127 <= a <= 127
    if np.less(np.amin(modified_image[:, :, 1]), ab_min) or np.greater(np.amax(modified_image[:, :, 1]), ab_max):
        print(f'a* value out of bound. min = {np.amin(modified_image[:, :, 1])}, max = {np.amax(modified_image[:, :, 1])}.')
        raise LabValueOutOfBound
    # -127 <= b <= 127
    if np.less(np.amin(modified_image[:, :, 2]), ab_min) or np.greater(np.amax(modified_image[:, :, 2]), ab_max):
        print('b* value out of bound.')
        raise LabValueOutOfBound
    
    return modified_image
