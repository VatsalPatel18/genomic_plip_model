import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import openslide
from PIL import Image
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import math
import random
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_dilation, disk
from concurrent.futures import ProcessPoolExecutor

class SlideProcessor:
    def __init__(self,img_processor=None, tile_size=1024, overlap=0, tissue_threshold=0.65, max_workers=30):
        self.tile_size = tile_size
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.max_workers = max_workers
        self.img_processor = img_processor

    def optical_density(self, tile):
        tile = tile.astype(np.float64)
        od = -np.log((tile+1)/240)
        return od

    def keep_tile(self, tile, tissue_threshold=None):
        if tissue_threshold is None:
            tissue_threshold = self.tissue_threshold
            
        if tile.shape[0:2] == (self.tile_size, self.tile_size):
            tile_orig = tile
            tile = rgb2gray(tile)
            tile = 1 - tile
            tile = canny(tile)
            tile = binary_closing(tile, disk(10))
            tile = binary_dilation(tile, disk(10))
            tile = binary_fill_holes(tile)
            percentage = tile.mean()

            check1 = percentage >= tissue_threshold

            tile = self.optical_density(tile_orig)
            beta = 0.15
            tile = np.min(tile, axis=2) >= beta
            tile = binary_closing(tile, disk(2))
            tile = binary_dilation(tile, disk(2))
            tile = binary_fill_holes(tile)
            percentage = tile.mean()

            check2 = percentage >= tissue_threshold

            return check1 and check2
        else:
            return False
        
    def filter_tiles(self, tile_indices, generator):
        def process_tile(tile_index):
            tile_size, overlap, zoom_level, col, row = tile_index
            tile = np.asarray(generator.get_tile(zoom_level, (col, row)))
            if self.keep_tile(tile, self.tissue_threshold):
                return col, row
            return None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(process_tile, tile_indices)
        
        return [result for result in results if result is not None]


    def get_tiles(self, filtered_tiles, tile_indices, generator):
        tiles = {}
        for col, row in filtered_tiles:
            # Find the tile_index with matching col and row
            tile_index = next((ti for ti in tile_indices if ti[3] == col and ti[4] == row), None)
            if tile_index:
                tile_size, overlap, zoom_level, col, row = tile_index
                tile = np.asarray(generator.get_tile(zoom_level, (col, row)))
                tiles[(col,row)] = tile
        return tiles

    def process_one_slide(self, file_loc):
        f2p = file_loc
        
        img1 = openslide.open_slide(f2p) 
        generator = DeepZoomGenerator(img1, tile_size=self.tile_size, overlap=self.overlap, limit_bounds=True)
        highest_zoom_level = generator.level_count - 1

        try:
            mag = int(img1.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            offset = math.floor((mag / 20) / 2)
            level = highest_zoom_level - offset
        except (ValueError, KeyError):
            level = highest_zoom_level

        zoom_level = level
        cols, rows = generator.level_tiles[zoom_level]
        tile_indices = [(self.tile_size, self.overlap, zoom_level, col, row) for col in range(cols) for row in range(rows)]
        
        filtered_tiles = self.filter_tiles(tile_indices, generator)

        if file_loc.endswith('.svs'):
            file = file_loc[-16:-4]
            print(file)
        
        tiles = self.get_tiles(filtered_tiles, tile_indices, generator)

        return tiles