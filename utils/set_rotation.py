#!/usr/bin/env python
# coding: utf-8

# # Face extraction
# 
# We use the MTCNN network in order to extract faces from pictures.
# 
# We will get all the faces that are found in a picture and return them.

from PIL import Image
import piexif
from os import listdir

def rotate_jpeg(filename):
    '''
    Rotate the image by its exif orientation tag value and remove the 
    orientation tag from the image's exif data.
    The new image is saved over the initial one.
    
    :param filename: (string) the path to the file
    :return: nothing
    '''
    img = Image.open(filename)
    if "exif" in img.info:
        exif_dict = piexif.load(img.info["exif"])

        if piexif.ImageIFD.Orientation in exif_dict["0th"]:
            orientation = exif_dict["0th"].pop(piexif.ImageIFD.Orientation)
            exif_bytes = piexif.dump(exif_dict)

            if orientation == 2:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                img = img.rotate(180)
            elif orientation == 4:
                img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 5:
                img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 6:
                img = img.rotate(-90, expand=True)
            elif orientation == 7:
                img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
            
            img.save(filename, exif=exif_bytes)
            
def rotate_directory(directory):
    '''
    Applies rotate_jpeg() to all the files in a directory.
    
    :param directory: (string) path of the directory 
    :return: nothing
    '''
    for filename in listdir(directory):
        rotate_jpeg(directory + '/' + filename)
