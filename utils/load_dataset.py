#!/usr/bin/env python
# coding: utf-8

# Load dataset
# 
# This is used to load the dataset from folders containing pictures. 
# The labels are created based on the names of the folders. 


from os import listdir
from os.path import isdir
from numpy import savez_compressed
from face_extract.py import extract_face
import imageio

