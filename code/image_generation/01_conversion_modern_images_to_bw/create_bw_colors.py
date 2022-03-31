from skimage import io, color, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import shutil

def convert_to_bw_lab_image(path):
    original_color_lab_image = color.rgb2lab(io.imread(path))
    bw_lab_picture = np.zeros_like(original_color_lab_image)
    bw_lab_picture[:,:, 0] = original_color_lab_image[:, :, 0]
    return bw_lab_picture

if __name__ == '__main__':

    directory_path = Path('image_generation/00_base_images/modern/tid2013/reference_images').absolute()
    export_path = Path('image_generation/01_conversion_modern_images_to_bw/export').absolute()
    image_file_names = os.listdir(directory_path)

    if export_path.is_dir():
        shutil.rmtree(export_path)
    os.makedirs(export_path)

    for image in image_file_names:
        new_file_name = f'{image.split(".")[0]}_bw.BMP'

        io.imsave(
            export_path / new_file_name,
            img_as_ubyte(color.lab2rgb(convert_to_bw_lab_image(directory_path / image))))
