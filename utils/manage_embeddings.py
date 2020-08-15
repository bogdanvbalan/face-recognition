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
from os import remove
import facenet
import tensorflow as tf
import matplotlib.pyplot as plt
from os import listdir

SAVE_FOLDER = getcwd() + '/data/datasets/embeddings/'

def get_embedding(model, img):
        ''' Get the face embedding for one face

        Args:
            model: The path to the pb file
            img: The path to the image file

        Raises: nothing

        Returns: yhat: the prediction of the model un the current face

        '''
        # load 
        face_pixels = plt.imread(img)

        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        
        # standardize pixel values
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        # transform into one sample
        samples = expand_dims(face_pixels, axis=0)

        # get embeddings
        with tf.Graph().as_default():
            with tf.Session() as sees:

                # load model
                facenet.load_model(model)

                # input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # fw pass to calculate embeddings
                feed_dict = { images_placeholder: samples, phase_train_placeholder: False}
                emb = sees.run(embeddings, feed_dict=feed_dict)

        return emb

def get_embedding_ex(model, img):
    ''' Get the face embedding for one face

    Args:
        model: The path to the pb file
        img: The path to the image file

    Raises: nothing

    Returns: yhat: the prediction of the model un the current face

    '''
    # load 
    #face_pixels = plt.imread(img)

    # scale pixel values
    face_pixels = img.astype('float32')
        
    # standardize pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # transform into one sample
    samples = expand_dims(face_pixels, axis=0)

    # get embeddings
    with tf.Graph().as_default():
        with tf.Session() as sees:

            # load model
            facenet.load_model(model)

            # input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # fw pass to calculate embeddings
            feed_dict = { images_placeholder: samples, phase_train_placeholder: False}
            emb = sees.run(embeddings, feed_dict=feed_dict)

    return emb

def save_embeddings(load_dir, save_dir, model):
    '''
    Calls get_embbeding in order to get the embbeding for 
    each picture. Saves an array of embbedings at the 
    path received.

    Args:

        load_dir (str): the path to the directory where the registration images are stored
        save_dir (str): the path to the npz file location where the 
                         embbedings will be saved
        model (str): the path to the model that will be used 
                      to get the embeddings

    Raises:

        nothing

    Returns:

        nothing
    '''
    print("Gettings embeddings for files in " + str(load_dir) + " and saving them to: " + str(save_dir))

    # get the embedding for each image
    embeddings = list()
    for filename in listdir(load_dir):
        embedding = get_embedding(model, load_dir + "\\" + filename)
        embeddings.append(embedding[0])

    embeddings = asarray(embeddings)

    # save the embeddings 
    savez_compressed(save_dir, embeddings)

