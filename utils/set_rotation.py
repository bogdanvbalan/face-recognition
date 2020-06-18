#!/usr/bin/env python
# coding: utf-8

# # Face extraction
# 
# We use the MTCNN network in order to extract faces from pictures.
# 
# We will get all the faces that are found in a picture and return them.

from PIL import Image
from PIL import ImageFile
import piexif
from os import listdir
from os.path import isdir

ImageFile.LOAD_TRUNCATED_IMAGES = True

def rotate_jpeg(filename, save_name, output_dir=None):
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
            print('Got to rotate_jpeg with ' + filename + ' and ' + output_dir)
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
            

            if output_dir == None:
                print('Got to save without outputdir')
                img.save(filename, exif=exif_bytes)
            else:
                print('Got to save to ',output_dir)
                img.save(output_dir + save_name, exif=exif_bytes)

        else:

            if output_dir != None:
                img.save(output_dir + save_name)
            
def rotate_directory(input_dir, output_dir=None):
    '''
    Applies rotate_jpeg() to all the files in a directory.
   
    Args:

        input_dir (str): path to the directory to be used as input
        output_dir (str): the path where the files should be saved,
                        if None they will be overwritten
    Raises:

        nothing

    Returns:

        nothing
    '''
    for filename in listdir(input_dir):
        current_path = input_dir + '/' + filename
        if isdir(current_path) == True:
            rotate_directory(current_path,output_dir)
        else:
            if filename[-1].lower() == 'g':
                rotate_jpeg(current_path,filename,output_dir=output_dir)
