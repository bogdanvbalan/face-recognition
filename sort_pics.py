import sys
import argparse
import facenet
from os import listdir
from numpy import load
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.set_rotation import rotate_directory
from utils.manage_embeddings import save_embeddings, get_embedding_ex
from utils.face_extract import extract_face

# main function
def main(args):
    # store the values from args
    imgs_dir = args.images_dir
    output_dir = args.output_dir
    model_path = args.model
    emb_path = args.reg_embbedings

    # check if the images need to be rotated
    if args.set_rotation:
        # we rotate the directory
        rotate_directory(imgs_dir)

    # load registered identity
    data = load(emb_path)
    identity_vec = data['arr_0']

    # we go over the images in the directory
    for filename in listdir(imgs_dir):
        # we extract all the faces in the current image
        current_img_path = imgs_dir + "\\" + filename
        faces = extract_face(current_img_path)
        for i in range(len(faces)):
            # compare each face that was extracted
            current_emb = get_embedding_ex(model_path, faces[i])
            embedding = asarray(current_emb)
            sum_tresh = sum(cosine_similarity(identity_vec, embedding))
            sum_tresh = sum_tresh/len(identity_vec)
            print("Sum tresh is: ", sum_tresh)
            if sum_tresh > 0.5:
                print("Got match in: ", filename)
                img = plt.imread(current_img_path)
                plt.imsave(output_dir + "\\" + filename, img)

# Parsing of the arguments
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images_dir', type=str, 
        help='Directory containing the images.')
    parser.add_argument('--output_dir', type=str,
        help='The path to the output directory.')
    parser.add_argument('--model', type=str,
        help='The path to the model (pb).')
    parser.add_argument('--reg_embbedings', type=str,
        help='The path to the registered identity embeddings.')
    parser.add_argument('--set_rotation', action="store_true",
        help='When set the images are rotated according to the default orientation')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
