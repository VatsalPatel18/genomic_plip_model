import os
import numpy as np
import pandas as pd
import torch
import pickle
import random
from sklearn.model_selection import train_test_split
import argparse

def process_split_and_save(data_dir, metadata_file, save_dir, args):

    df = pd.read_csv(metadata_file)
    df = df.set_index('HNSC')
    there = set(df.index)
    wsi_there = os.listdir(data_dir)
    use = list(there.intersection(wsi_there))
    df = df.loc[use]
    df['cluster'] = df['cluster'] - 2

    df = df.sample(frac=1)

    class1 = list(df[df['cluster'] == 1].index)
    class0 = list(df[df['cluster'] == 0].index)

    C1_X_train, C1_X_test = train_test_split(class1, test_size=args.test_val_size)
    C0_X_train, C0_X_test = train_test_split(class0, test_size=args.test_val_size)

    C1_X_validate, C1_X_test = train_test_split(C1_X_test, test_size=args.val_size)
    C0_X_validate, C0_X_test = train_test_split(C0_X_test, test_size=args.val_size)


    X_train = C1_X_train + C0_X_train
    X_test = C1_X_test + C0_X_test
    X_validate = C1_X_validate + C0_X_validate

    random.shuffle(X_train)
    random.shuffle(X_test)
    random.shuffle(X_validate)

    data_info = {'train': X_train, 'test': X_test, 'validate': X_validate}
    with open(os.path.join(save_dir, 'data_info.pkl'), 'wb') as f:
        pickle.dump(data_info, f)

    data = {'train': {'X': [], 'Y': []}, 'test': {'X': [], 'Y': []}, 'validate': {'X': [], 'Y': []}}
    for phase in ['train', 'validate', 'test']:
        for pID in data_info[phase]:
            fol_p = os.path.join(data_dir, pID)
            tiles = os.listdir(fol_p)
            tile_data = []
            for tile in tiles:
                tile_p = os.path.join(fol_p, tile)
                np1 = torch.load(tile_p).numpy()
                tile_data.append(np1)

            data[phase]['X'].extend(tile_data)
            data[phase]['Y'].extend([df.loc[pID] for _ in range(len(tile_data))])

        data[phase]['X'] = np.squeeze(np.array(data[phase]['X']), axis=1)
        data[phase]['Y'] = np.array(data[phase]['Y'])

    with open(os.path.join(save_dir, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)
        
    wsi_data = {}
    for pID in df.index:  # using df after it has been filtered for 'use'
        fol_p = os.path.join(data_dir, pID)
        tiles = os.listdir(fol_p)
        tile_data = []
        for tile in tiles:
            tile_p = os.path.join(fol_p, tile)
            tile_data.append(torch.load(tile_p).numpy())

        np1 = np.array(tile_data)
        wsi_data[pID] = {'tiles': np.squeeze(np1, axis=1), 'class': df.loc[pID]['cluster']}

    with open(os.path.join(save_dir, 'wsi_data.pkl'), 'wb') as f:
        pickle.dump(wsi_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split GPLIP tile features into training, testing, and validation sets.')
    parser.add_argument('--data_dir', type=str, default='kaal_extract', help='Directory containing the extracted features.')
    parser.add_argument('--metadata_file', type=str, default='g2_g3.csv', help='CSV file containing the metadata for the samples.')
    parser.add_argument('--save_dir', type=str, default='Datasets', help='Directory to save the split data.')
    parser.add_argument('--test_val_size', type=float, default=0.3, help='Size of the test/validation set combined.')
    parser.add_argument('--val_size', type=float, default=0.4, help='Size of the validation set from the test/validation split.')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    process_split_and_save(args.data_dir, args.metadata_file, args.save_dir, args)

