import torch
from transformers import CLIPVisionModel

class GenomicPLIPModel(torch.nn.Module):
    def __init__(self, original_model):
        super(GenomicPLIPModel, self).__init__()
        self.vision_model = original_model.vision_model
        self.vision_projection = torch.nn.Linear(768, 512)
        self.fc_layer = torch.nn.Linear(4, 512)  # Fully connected layer for the 4D vector

    def forward(self, pixel_values, score_vector):
        vision_output = self.vision_model(pixel_values)
        pooled_output = vision_output.pooler_output
        vision_features = self.vision_projection(pooled_output)
        score_features = self.fc_layer(score_vector)
        
        return vision_features, score_features
