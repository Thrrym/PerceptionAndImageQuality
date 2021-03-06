#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import random
import re
from PIL import Image


### Generate Picture Files ###

picture_path = "Pictures/"

pictures_file = []

max_height = 300
max_width = 600


# Takes an Str List and sorts it
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


# Goes trough the Pictures Folder, resizes every picture to 25% and adds it to our result (random order)
def getPictures():
    dirs = os.listdir(picture_path)
    for group in dirs:
        print(group)
        if group == "README.md":
            break
        picturedir = sorted_alphanumeric(os.listdir(picture_path + group))
        pictures = []
        buntheit = 0
        for picture in picturedir:
            print()
            if buntheit == 9:
                continue
            path_help = picture_path + group + "/"
            resize(path_help, picture)
            pictures.append([buntheit, picture_path + group + "/" + picture.split("_")[1] + ".jpg"])
            buntheit += 1

        random.shuffle(pictures)
        pictures_file.append(pictures)
    random.shuffle(pictures_file)
    print(pictures_file)

    with open("Pictures/pictures.csv", "w", newline="\n") as f:
        write = csv.writer(f)
        write.writerows(pictures_file)


def resize(path, image):
    img = Image.open(path + image)
    width, height = img.size
    ratio = min(max_width / width, max_height / height)
    """
    if height > 700:
        ratio = 0.25
    """
    img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.ANTIALIAS)
    '''
    if width < 600:
        img = img.resize((int(img.size[0] * 0.5), int(img.size[1] * 0.5)), Image.ANTIALIAS)
    elif height > 700 or width > 700:
        img = img.resize((int(img.size[0] * 0.25), int(img.size[1] * 0.25)), Image.ANTIALIAS)
    else:
        img = img.resize((int(img.size[0] * 0.3), int(img.size[1] * 0.3)), Image.ANTIALIAS)
    # print(path + str(buntheit) + ".jpg")
    '''
    img.save(path + image.split("_")[1] + ".jpg")


if __name__ == '__main__':
    getPictures()

