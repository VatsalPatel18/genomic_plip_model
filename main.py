import torch
import numpy as np
import argparse
from classes.slide_processor import SlideProcessor
from classes.genomic_plip_model import GenomicPLIPModel 
from classes.binary_neural_classifier import SimpleNN
from transformers import CLIPImageProcessor

def main(svs_file_path, tile_size, max_workers):
    # Initialize SlideProcessor and process the slide
    processor = SlideProcessor(tile_size=tile_size, overlap=0, max_workers=max_workers)
    tiles = processor.process_one_slide(svs_file_path)

    # Assuming you have a method to concatenate or batch process these tiles
    image_array = np.stack(list(tiles.values()))

    # Load the pre-trained GenomicPLIPModel and preprocess the tiles
    clip_processor = CLIPImageProcessor.from_pretrained("./genomic_plip_model")
    pro_data = np.array(clip_processor(image_array)['pixel_values'])

    # Load the GenomicPLIPModel and make predictions
    gmodel = GenomicPLIPModel.from_pretrained("./genomic_plip_model")
    gmodel.eval()
    pred_data = gmodel(torch.from_numpy(pro_data))

    # Load the SimpleNN model and make predictions
    device = torch.device("cpu")
    model = SimpleNN().to(device)
    model.load_state_dict(torch.load('./models/classifier.pth', map_location=torch.device('cpu')))
    model.eval()

    # Get the mean of the predictions
    output = model(pred_data).mean()

    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("svs_file_path", type=str, help="Path to the SVS file")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size for processing the image")
    parser.add_argument("--max_workers", type=int, default=32, help="Number of workers for processing the tiles")

    args = parser.parse_args()

    result = main(args.svs_file_path, args.tile_size, args.max_workers)
    print(f"Predicted value: {result}")
