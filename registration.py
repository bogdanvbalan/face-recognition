from utils.manage_embeddings import save_embeddings, get_embedding
import sys
import argparse

# main function
def main(args):

    # call save_embeddings in order to register the new identity
    save_embeddings(args.images_dir, args.output_file, args.model)

# Parsing of the arguments
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--images_dir', type=str, 
        help='Directory containing the registration images.')
    parser.add_argument('--output_file', type=str,
        help='The name of the output file (embeddings vectors).')
    parser.add_argument('--model', type=str,
        help='The path to the model (pb).')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))