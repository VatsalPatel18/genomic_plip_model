import os
import random
import shutil
from sklearn.model_selection import train_test_split
import argparse

def main(num_tiles_per_patient, source_dir, dataset_dir ,test_val_size, val_size):
    # Create necessary directories if they don't exist
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'validate'), exist_ok=True)

    # List all patient directories
    files = os.listdir(source_dir)

    # Split the data into train, test, and validation sets
    train, test_val = train_test_split(files, test_val_size)
    test, val = train_test_split(test_val, val_size)

    # Function to process and copy files
    def process_and_copy(file_list, type):
        for file in file_list:
            fol_p = os.path.join(source_dir, file)
            tiles = os.listdir(fol_p)
            selected_tiles = random.sample(tiles, min(num_tiles_per_patient, len(tiles)))
            for tile in selected_tiles:
                tile_p = os.path.join(fol_p, tile)
                new_p = os.path.join(dataset_dir, type, tile)
                shutil.copy(tile_p, new_p)

    # Process and copy files for each dataset
    process_and_copy(train, 'train')
    process_and_copy(test, 'test')
    process_and_copy(val, 'validate')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into train, test, and validation sets.')
    parser.add_argument('--num_tiles_per_patient', type=int, default=595,
                        help='Number of tiles to select per patient.')
    parser.add_argument('--source_dir', type=str, default='plip_preprocess',
                        help='Directory containing patient folders.')
    parser.add_argument('--dataset_dir', type=str, default='Datasets/train_03',
                        help='Root directory for the train, test, and validate directories.')
    parser.add_argument('--test_val_size', type=float, default=0.4,
                        help='Size of the test and validation sets combined.')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='Proportion of validation set in the test-validation split.')

    args = parser.parse_args()
    
    main(args.num_tiles_per_patient, args.source_dir, args.dataset_dir,args.test_val_size, args.val_size)
    
    
