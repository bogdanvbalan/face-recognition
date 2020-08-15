import os
import subprocess
import sys

# Train Params
python_loc = r"C:\Users\John\Anaconda3\envs\visual_studio_env\python.exe"                                # python exe location
compare_script = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\registration.py"        # train script location
model = r"train_10000es_8epcs.pb"
img_folder  = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\test\train\bogdan"
output_file = r"train_10000es_8epcs.npz"

def test_compare():
    print('test_compare')
    argv = [python_loc,
            compare_script,
            "--images_dir", img_folder,
            "--output_file", output_file,
            "--model", model
            ]
    subprocess.call(argv)

imgs_dir = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\test2\input"
output_dir = r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\test2\output"
sort_script= r"D:\Projects\Face_Recognition\Face_Recognition\Face-Recognition\sort_pics.py"  

def abc():
    print('test_compare')
    argv = [python_loc,
            sort_script,
            "--images_dir", imgs_dir,
            "--output_dir", output_dir,
            "--model", model,
            "--reg_embbedings", output_file,
            '--set_rotation',
            ]
    subprocess.call(argv)

abc()



