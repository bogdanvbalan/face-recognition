import os
import base64
import struct
import numpy as np
import cv2

# fisierele de input si output 
input_file = open("D:\\uTorrent\\MS-Celeb-1M\\data\\aligned_face_images\\FaceImageCroppedWithAlignment.tsv", "r", encoding='utf-8')
output_folder = './MsCeleb/'

# Script folosit pentru a extrage pozele din baza de date
while True:
  line = input_file.readline()
  if line:
    data_info = line.split('\t')
    filename = data_info[0] + "/" + data_info[1] + "_" + data_info[4] + ".jpg"

    img_dec_string = base64.b64decode(data_info[6])
    img_data = np.fromstring(img_dec_string, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (160,160), interpolation = cv2.INTER_AREA)
    output_file_path = output_folder + "/" + filename 
    if os.path.exists(output_file_path):
      print( output_file_path + " exists")

    output_path = os.path.dirname(output_file_path)
    if not os.path.exists(output_path):
      os.mkdir(output_path)

    img_file = open(output_file_path, 'w')
    cv2.imwrite(output_file_path, img)
    img_file.close()
  else:
    break

input_file.close