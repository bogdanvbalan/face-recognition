#!/usr/bin/env python
# coding: utf-8

# Get embeddings
# 
# This is used to get the embbedings for one or multiple
# faces

from os import getcwd
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

SAVE_FOLDER = getcwd() + '/data/datasets/embeddings/'

def get_embedding(model, face_pixels):
        ''' Get the face embedding for one face

        Args:
            model: A nn model that is loaded via load_model
            face_pixels: An array containing the face

        Raises: nothing

        Returns: yhat: the prediction of the model un the current face

        '''
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        
        # standardize pixel values
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        # transform into one sample
        samples = expand_dims(face_pixels, axis=0)

        # make prediction
        yhat = model.predict(samples)

        return yhat[0]

def save_embeddings(load_dir, save_dir, model):
    '''
    Calls get_embbeding in order to get the embbeding for 
    each picture. Saves an array of embbedings at the 
    path received.

    It is assumed that the data file contains is split in 4 
    sets (X and y for both train and test)
    Args:

        load_dir (str): the path to the npz file that contains the 
                        pictures
        save_dir (str): the path to the npz file location where the 
                         embbedings will be saved
        model (str): the path to the model that will be used 
                      to get the embeddings

    Raises:

        nothing

    Returns:

        nothing
    '''
    print('save_embeddings() - Load dir', load_dir)

    data = load(load_dir)
    x_train, y_train, x_test, y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    model = load_model(model)
    model.summary()

    new_x_train = list()
    for face_pixels in x_train:
        embedding = get_embedding(model, face_pixels)
        new_x_train.append(embedding)
    
    new_x_train = asarray(new_x_train)

    new_x_test = list()
    for face_pixels in x_test:
        embedding = get_embedding(model, face_pixels)
        new_x_test.append(embedding)

    new_x_test = asarray(new_x_test)

    savez_compressed(save_dir, new_x_train, y_train, new_x_test, y_test)

