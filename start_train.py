import subprocess
import os

# Train Params
python_loc = r"C:\Users\John\Anaconda3\envs\visual_studio_env\python.exe"                                # python exe location
train_script = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\train_softmax.py"        # train script location
logs_dir = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\logs\facenet"                # location where logs are saved
models_dir = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\models\facenet"            # location where the models will be saved
data_dir = r"D:\Projects\MSCelebTest"                                                                    # directory containing the training set
image_size = '160'                                                                                       # size of images in training set (image_size x image_size)
model_type = 'models.inception_resnet_v2'                                                                # the model that will be used for training
optimizer = 'ADAM'                                                                                       # optimizer used in training
learning_rate = '0.01'                                                                                   # learning rate used in traning, if set to -1 it will use the values in scheduler
batch_size = '32'                                                                                         # number of images in each batch
max_nrof_epochs = '2'                                                                                    # number of epochs to run 
keep_prob = '0.8'                                                                                        # keep probability for dropout
lr_file = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\data\lr_params.txt"           # location of learning rate scheduler file (case when lr is -1)
emb_size = '512'                                                                                         # size of embeddings vector
val_split = '0.2'                                                                                        # part of the training set used for validation
val_epcohs = '1'                                                                                         # number of epochs at which validation is done
epoch_size = '10'                                                                                        # number of batches per epoch

# Here is where you can add/modify/remove arguments for training
# The train is done using softmax on Resnet Inception V2
def train_softmax_inception_v2():
    print('Training using Softmax on Inception Resnet V2')
    argv = [python_loc,
            train_script,
            '--logs_dir', logs_dir,
            '--models_dir', models_dir,
            '--data_dir', data_dir,
            '--image_size', image_size,
            '--model_def', model_type,
            '--optimizer', optimizer,
            '--learning_rate', learning_rate,
            '--max_nrof_epochs', max_nrof_epochs,
            '--keep_probability', keep_prob,
            '--learning_rate_schedule_file', lr_file,
            '--embedding_size', emb_size,
            '--validation_set_split_ratio', val_split,
            '--validate_every_n_epochs', val_epcohs,
            '--epoch_size', epoch_size,
            '--batch_size', batch_size
            ]
    subprocess.call(argv, shell=True)

train_softmax_inception_v2()
