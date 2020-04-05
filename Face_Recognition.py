from utils import face_extract, set_rotation
import matplotlib.pyplot as plt
from os import listdir

set_rotation.rotate_directory('pics')

for item in listdir('pics'):
    faces = face_extract.extract_face('pics/' + item)

    for face in faces:
        plt.imshow(face)
        plt.show()

