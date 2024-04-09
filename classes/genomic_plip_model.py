import torch
from transformers import PreTrainedModel, CLIPConfig, CLIPModel

class GenomicPLIPModel(PreTrainedModel):
    config_class = CLIPConfig 

    def __init__(self, config):
        super(GenomicPLIPModel, self).__init__(config)
        vision_config = CLIPModel.config_class.from_pretrained('openai/clip-vit-base-patch32')
        self.vision_model = CLIPModel(vision_config).vision_model
        self.vision_projection = torch.nn.Linear(768, 512)

    def forward(self, pixel_values):
        vision_output = self.vision_model(pixel_values)
        pooled_output = vision_output.pooler_output
        vision_features = self.vision_projection(pooled_output)
        
        return vision_features
