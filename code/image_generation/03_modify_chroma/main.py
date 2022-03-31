#import sys
#import logging
#import readline
import numpy as np
import os
import matplotlib.pyplot as plt
from shutil import rmtree
from pathlib import Path
from skimage import io, color, img_as_ubyte
from chroma_utils import get_max_chroma_factor, get_factors_between_min_max, modify_lab_image_chroma

#logging.basicConfig(level=logging.ERROR)

def save_chroma_factors_plot(factors: list, path):
    x = np.arange(len(factors))
    plt.scatter(x, factors)
    plt.xlabel('Modification number')
    plt.ylabel('Chroma factor')
    plt.yticks(np.around(factors, 1))
    plt.title('Chroma modification factors')
    plt.savefig(path, facecolor='white')
    plt.close('all')

def save_overview(images, factors, file_id, path):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(ncols=3, nrows=3, figure=fig)
    
    num = 0
    for x in [0, 1, 2]:
        for y in [0, 1, 2]:
            ax = fig.add_subplot(gs[x, y])
            ax.imshow(images[num])
            ax.set_title(f'Factor {factors[num]}')
            ax.set_yticks([])
            ax.set_xticks([])
            num += 1
    fig.suptitle(f'Image {file_id} chroma factor overview')
    plt.savefig(path, facecolor='white', dpi=300)
    plt.close('all')

def create_images_with_factors(file_name):
    #if len(sys.argv) == 1:
    #    logging.error('No arguments given.')
    #    return

    #file_name = sys.argv[1]
    file_path = Path('image_generation/02_recolor/export').absolute() / file_name

    if not file_path.is_file():
        #logging.error(f'No file: {file_name}.')
        #return
        raise FileNotFoundError
    
    original_color_lab_image = color.rgb2lab(io.imread(file_path))
    max_factor = get_max_chroma_factor(original_color_lab_image)

    factors = get_factors_between_min_max(
        max_value=max_factor,
        min_value=0.6,
        plot_path= ''
    )
    factors = np.round(factors, 2)
    modified_images = []
    for factor in factors:
        modified_images.append(img_as_ubyte(color.lab2rgb(modify_lab_image_chroma(factor, original_color_lab_image))))

    export_dir = Path('image_generation/04_completed_images').absolute() / file_name

    if export_dir.is_dir():
        rmtree(export_dir)
    os.makedirs(export_dir)

    for image, factor in zip(modified_images, factors):
        io.imsave(
            export_dir / f'{file_name.split("_")[0]}_{np.around(factor, 2)}_chroma.JPG',
            image
        )
        save_chroma_factors_plot(factors, export_dir / f'{file_name.split("_")[0]}_chroma_factors.png')

    save_overview(modified_images, factors, file_name.split("_")[0], export_dir / f'{file_name.split("_")[0]}_overview.png')

def main():
    directory_path = Path('image_generation/02_recolor/export').absolute()
    export_path = Path('image_generation/04_completed_images').absolute()

    if export_path.is_dir():
        rmtree(export_path)
    os.makedirs(export_path)

    image_file_names = os.listdir(directory_path)
    image_file_names = list(filter(lambda x: x.split('_')[1].endswith('recolor.BMP'), image_file_names))

    for image in image_file_names:
        create_images_with_factors(image)

if __name__ == '__main__':
    main()