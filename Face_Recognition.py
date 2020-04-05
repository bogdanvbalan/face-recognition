
from numpy import savez_compressed
from utils.face_extract import extract_face
from utils.set_rotation import rotate_directory
from utils.load_dataset import load_dataset



x_train, y_train = load_dataset('pics/train/',preprocessing=1)
print(x_train.shape)
print(y_train.shape)

x_test, y_test = load_dataset('pics/val/',preprocessing=0)
print(x_test.shape)
print(y_test.shape)

savez_compressed('custom_dataset.npz', x_train, y_train, x_test, y_test)