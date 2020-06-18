from os import getcwd
from utils.manage_dataset import load_and_save_dataset
from utils.manage_embeddings import save_embeddings, get_embedding
from numpy import asarray
from numpy import sum
import matplotlib.pyplot as plt
from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from keras.models import load_model
from os import remove
from sklearn import metrics
import facenet

CURRENT_PATH = getcwd()
SAVE_DIR = CURRENT_PATH + '/data/datasets/'
RAW_LOCATION = SAVE_DIR + '/raw/'
EMBEDDINGS_LOCATION = SAVE_DIR + '/embeddings/'
MODEL_LOCATION = CURRENT_PATH + '/data/model/saved/'
TEST_DIM = 4542

model = facenet.load_model('abc.pb')
model.summary()
# Load and save the dataset in npz format
#load_and_save_dataset('test', RAW_LOCATION + 'custom_dataset_single.npz', 1, 0)

# Get and save the embeddings in npz format
save_embeddings(RAW_LOCATION + 'custom_dataset_single.npz', EMBEDDINGS_LOCATION + 'custom_embeddings_single.npz', model)

# load dataset
data = load(EMBEDDINGS_LOCATION + 'custom_embeddings_single.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
#in_encoder = Normalizer(norm='l2')
#trainX = in_encoder.transform(trainX)
#testX = in_encoder.transform(testX)

#print(trainX.shape)
#print(trainX[0].shape)

#print(testX.shape)
#print(testX[0].shape)

#sample1 = asarray(trainX[0])
#sample2 = asarray(testX[300])

#sum_tresh = sum(cosine_similarity(trainX, [sample2]))
#print(sum_tresh/7)

#wrong_pics = []

#for i in range(295):
#    sample = asarray(testX[i])
#    sum_tresh = sum(cosine_similarity(trainX, [sample]))
#    sum_tresh = sum_tresh/7
#    if sum_tresh < 0.5:
#        print('Incorrect at i = ', i)
#        wrong_pics.append(i)

#for i in range(295,590):
#    sample = asarray(testX[i])
#    sum_tresh = sum(cosine_similarity(trainX, [sample]))
#    sum_tresh = sum_tresh/7
#    if sum_tresh > 0.5:
#        print('Incorrect at i = ', i)
#        wrong_pics.append(i)

#data = load(RAW_LOCATION + 'custom_dataset_single.npz')
#trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

#for item in wrong_pics:
#    plt.imshow(testX[item])
#    plt.show()

model = load_model('abc.pb')

#for filename in listdir('extracted'):
#    current_im = plt.imread('extracted/' + filename)
#    embedding = get_embedding(model, current_im)
#    embedding = asarray(embedding)
#    sum_tresh = sum(cosine_similarity(trainX, [embedding]))
#    sum_tresh = sum_tresh/7
#    if sum_tresh > 0.5:
#        plt.imsave('custom_pics/bogdan/' + filename,current_im)
#        remove('extracted/' + filename)

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0

for filename in listdir('custom_pics/bogdan'):
    current_im = plt.imread('custom_pics/bogdan/' + filename)
    embedding = get_embedding(model, current_im)
    embedding = asarray(embedding)
    sum_tresh = sum(cosine_similarity(trainX, [embedding]))
    sum_tresh = sum_tresh/7
    if sum_tresh > 0.5:
        true_pos +=1
    else:
        false_pos +=1

for filename in listdir('custom_pics/other'):
    current_im = plt.imread('custom_pics/other/' + filename)
    embedding = get_embedding(model, current_im)
    embedding = asarray(embedding)
    sum_tresh = sum(cosine_similarity(trainX, [embedding]))
    sum_tresh = sum_tresh/7
    if sum_tresh > 0.5:
        false_neg +=1
    else:
        true_neg +=1

print("True positive: ", true_pos)
print("False positive: ", false_pos)
print("False negative: ", false_neg)
print("True negative: ", true_neg)
print("Accuracy is: ", ((true_pos + true_neg)/TEST_DIM) * 100)
