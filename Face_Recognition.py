from utils import load_dataset
import matplotlib.pyplot as plt
from os import listdir
from utils.face_extract import extract_face
from utils.set_rotation import rotate_directory

rotate_directory('pics')

for item in listdir('pics'):
    faces = extract_face('pics/' + item)

    for face in faces:
        plt.imshow(face)
        plt.show()

