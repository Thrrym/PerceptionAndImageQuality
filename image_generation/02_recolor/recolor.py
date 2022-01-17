
from skimage import io, color, img_as_ubyte
from colorization.colorizers import *
from pathlib import Path
import os, sys, pickle
import matplotlib.pyplot as plt
import shutil

from chroma_utils import save_plot_lab_image_structure

def get_average_chroma_value(lab_img) -> float:
    """ Get average chroma value over the entire image. """
    a_2 = np.power(lab_img[:, :, 1], 2)
    b_2 = np.power(lab_img[:, :, 2], 2)
    return np.median(np.sqrt(a_2 + b_2))

def recolor(path_to_directory, export_path, colorizer) -> list:
    """ Recolor a black and white picture with pre-trained ML algorithm. """
    image_file_names = os.listdir(path_to_directory)
    average_chroma_values = []

    for image in image_file_names:
        print(image)
        new_file_name = f'{image.split("_")[0]}_recolor.BMP'
        img = load_img(path_to_directory / image)
        
        # Recolor steps:
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        out_img = postprocess_tens(tens_l_orig, colorizer(tens_l_rs).cpu())

        # Save image and structure plots.
        io.imsave(export_path / new_file_name, img_as_ubyte(out_img))
        lab_converted_image = color.rgb2lab(out_img)
        save_plot_lab_image_structure(
            lab_image=lab_converted_image,
            title=new_file_name,
            path=export_path / f'{image.split("_")[0]}_structure.png')
        
        # Store average chroma value of the recolored image.
        average_chroma_values.append([new_file_name, get_average_chroma_value(lab_converted_image)])
    
    return average_chroma_values

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

    average_chroma_values = []
    for path in directory_paths:
        average_chroma_values += recolor(path, export_path=export_path, colorizer=colorizer_siggraph17)

    pkl_path = Path('image_generation/02_recolor/average_chroma_values.pkl').absolute()
    with open(pkl_path, mode='wb') as pkl_file:
        pickle.dump(average_chroma_values, pkl_file)