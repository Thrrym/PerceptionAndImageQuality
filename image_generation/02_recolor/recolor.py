
from skimage import io, color, img_as_ubyte
from colorization.colorizers import *
from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import shutil

from chroma_utils import save_plot_lab_image_structure

def recolor(path_to_directory, export_path, colorizer):
    image_file_names = os.listdir(path_to_directory)

    for image in image_file_names:
        print(image)
        new_file_name = f'{image.split("_")[0]}_recolor.BMP'
        img = load_img(path_to_directory / image)
        #if img.shape[-1] == 4:
        #    img = color.rgba2rgb(img)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        out_img = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())
        io.imsave(export_path / new_file_name, img_as_ubyte(out_img))

        save_plot_lab_image_structure(
            lab_image=color.rgb2lab(out_img),
            title=new_file_name,
            path=export_path / f'{image.split("_")[0]}_structure.png')


if __name__ == '__main__':
    directory_paths = [
        Path('image_generation/01_conversion_modern_images_to_bw/export').absolute(),
        Path('image_generation/00_base_images/historic').absolute()
    ]
    export_path = Path('image_generation/02_recolor/export').absolute()

    # Remove old files and create empty export directory.
    if export_path.is_dir():
        shutil.rmtree(export_path)
    os.makedirs(export_path)

    # Get the machine learning model.
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()

    for path in directory_paths:
        recolor(path, export_path=export_path, colorizer=colorizer_siggraph17)