from os import getcwd
from utils.manage_dataset import load_and_save_dataset
from utils.manage_embeddings import save_embeddings
from numpy import asarray
from numpy import sum
import matplotlib.pyplot as plt
from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity



CURRENT_PATH = getcwd()
SAVE_DIR = CURRENT_PATH + '/data/datasets/'
RAW_LOCATION = SAVE_DIR + '/raw/'
EMBEDDINGS_LOCATION = SAVE_DIR + '/embeddings/'
MODEL_LOCATION = CURRENT_PATH + '/data/model/saved/'

# Load and save the dataset in npz format
#load_and_save_dataset('test', RAW_LOCATION + 'custom_dataset_single.npz', 1, 0)

# Get and save the embeddings in npz format
#save_embeddings(RAW_LOCATION + 'custom_dataset_single.npz', EMBEDDINGS_LOCATION + 'custom_embeddings_single.npz', MODEL_LOCATION + 'facenet_keras.h5')

# load dataset
data = load(EMBEDDINGS_LOCATION + 'custom_embeddings_single.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
#in_encoder = Normalizer(norm='l2')
#trainX = in_encoder.transform(trainX)
#testX = in_encoder.transform(testX)

print(trainX.shape)
print(trainX[0].shape)

print(testX.shape)
print(testX[0].shape)

sample1 = asarray(trainX[0])
sample2 = asarray(testX[300])

sum_tresh = sum(cosine_similarity(trainX, [sample2]))
print(sum_tresh/7)

wrong_pics = []

for i in range(295):
    sample = asarray(testX[i])
    sum_tresh = sum(cosine_similarity(trainX, [sample]))
    sum_tresh = sum_tresh/7
    if sum_tresh < 0.5:
        print('Incorrect at i = ', i)
        wrong_pics.append(i)

for i in range(295,590):
    sample = asarray(testX[i])
    sum_tresh = sum(cosine_similarity(trainX, [sample]))
    sum_tresh = sum_tresh/7
    if sum_tresh > 0.5:
        print('Incorrect at i = ', i)
        wrong_pics.append(i)

data = load(RAW_LOCATION + 'custom_dataset_single.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

for item in wrong_pics:
    plt.imshow(testX[item])
    plt.show()