from os import getcwd
from utils.manage_dataset import load_and_save_dataset
from utils.manage_embeddings import save_embeddings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report, confusion_matrix

CURRENT_PATH = getcwd()

# Load and save the dataset in npz format
#load_and_save_dataset('pics',1,0)

# Get and save the embeddings in npz format
#save_embeddings(CURRENT_PATH + '/data/datasets/raw/custom_dataset.npz', CURRENT_PATH + '/data/model/saved/facenet_keras.h5')

# load dataset
data = load(CURRENT_PATH + '/data/datasets/embeddigns/custom_embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

print(trainX.shape)
print(trainX[0].shape)
print(trainX[0])

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX,trainy)

# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)

# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

print(confusion_matrix(testy,yhat_test))
print(classification_report(testy,yhat_test))

svc_disp = plot_roc_curve(model, testX, testy)
plt.show()