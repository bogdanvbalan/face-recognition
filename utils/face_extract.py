#!/usr/bin/env python
# coding: utf-8

# # Face extraction
# 
# We use the MTCNN network in order to extract faces from pictures.
# 
# We will get all the faces that are found in a picture and return them.


from mtcnn import MTCNN
from PIL import Image
from numpy import asarray

def extract_face(filename, required_size=(160,160)):
    '''
    Extract a single face from a picture using the 
    mtcnn network.
    
    :param filename: (string) the path to the file
    :return: nothing
    '''
    try:
        # load image form file and convert to RGB
        print(filename)
        image = Image.open(filename)
        image = image.convert('RGB')
        
        # convert to array
        pixels = asarray(image)
        
        # create an instance of mtcnn using default
        # weights
        detector = MTCNN()
        
        # get the faces in the image
        results = detector.detect_faces(pixels)
        faces_no = len(results)
        
        # extract all the faces in the image
        faces_array = []
        
        for i in range(faces_no):
            # get the coordinates of the bounding box
            x1, y1, width, height = results[i]['box']
            x1, y1 = abs(x1), abs(y1) # avoid known issue
            x2, y2 = x1 + width, y1 + height
            
            # get the actual part of the image
            face = pixels[y1:y2, x1:x2]
            
            # resize to the required size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            image = asarray(image)
            faces_array.append(image)
        
        faces_array = asarray(faces_array)
        return faces_array
    
    except IndexError:
        
        print('IndexError: ' + filename)
        return None

