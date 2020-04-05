#!/usr/bin/env python
# coding: utf-8

# Manage dataset
# 
# This is used to load the dataset from folders containing pictures. 
# The labels are created based on the names of the folders. 

from os import getcwd
from os import listdir
from os.path import isdir
from numpy import asarray
from numpy import savez_compressed
from utils.face_extract import extract_face
from utils.set_rotation import rotate_directory
import imageio

def load_faces(directory):
        ''' Use extract_faces() in order to get the faces from a specific 
              directory

        Args:
            directory (string): The path to the directory

        Raises: nothing

        Returns: faces: a list containing the extracted faces

        '''
        faces = list()

        # go over each item in the directory and call extract_face
        for filename in listdir(directory):
          path = directory + filename

          extracted_faces = extract_face(path)

          if faces is not None:
            #imageio.imwrite('/content/drive/My Drive/Colab Notebooks/Face Recognition/test2/val/extracted/' + filename,face)
            for face in extracted_faces:
                faces.append(face)

        return faces

def load_dataset(directory, preprocessing=0):
        ''' Use load_faces() in order to get the faces from a specific 
              directory and call this for multiple directories
            Apply preprocessing according to the value received:
            0 - no preprocessing
            1 - extract faces
            2 - set rotation and extract faces

        Args:
            directory (string): The path to the main directory

        Raises: nothing

        Returns: X, y: the training examples and the labels

        '''
        assert preprocessing >= 0,"The value of preprocessing should be 0, 1 or 2"
        assert preprocessing <= 3,"The value of preprocessing should be 0, 1 or 2"

        X, y = list(), list()

        # go through each subdirectory
        for subdir in listdir(directory):
          path = directory + subdir + '/'

          # check if it is a directory
          if not isdir(path):
            continue
          else:
              if preprocessing == 2:
                  rotate_directory(path)
                  
          # load the faces in the directory
          if preprocessing != 0:
            print('Received a call to load_faces() for ' + path)
            faces = load_faces(path)
          else:
              print('Manually loading the imgs in ' + path)
              faces = list()
              for filename in listdir(path):
                  current_file = path + filename
                  img = imageio.imread(current_file)
                  faces.append(img)
            
          #  create labels
          labels = [subdir for _ in range(len(faces))]

          print('>loaded %d examples for class: %s' % (len(faces),subdir))

          X.extend(faces)
          y.extend(labels)

        return asarray(X), asarray(y)

def load_and_save_dataset(load_directory, save_directory, preprocess_train=0, preprocess_test=0):
        ''' Use load_dataset() in order to load and save the dataset
              found in the directory received as arguments
            Apply preprocessing according to the value received:
            0 - no preprocessing
            1 - extract faces
            2 - set rotation and extract faces

            The dataset is saved in /data/datasets/raw

        Args:

            load_directory (string): The path to the directory that contains pictures
            save_directory (string): The path to the folder where the array is saved
            preprocess_train (int): Described above
            preprocess_test (int): Described above

        Raises: nothing

        Returns: nothing

        '''
        x_train, y_train = load_dataset(load_directory + '/train/',preprocessing=preprocess_train)

        x_test, y_test = load_dataset(load_directory + '/val/',preprocessing=preprocess_test)

        savez_compressed(save_directory, x_train, y_train, x_test, y_test)