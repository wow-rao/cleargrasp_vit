#
# cleargrasp_vit/models/vit_dense_prediction.py
#
from vit_core import VisionTransformer


def create_vit_dense_predictor(model_config: dict, output_channels: int) -> VisionTransformer:
    """Helper function to create a ViT dense prediction model from a config dict."""
    vit_encoder = VisionTransformer(output_channels=output_channels, **model_config)
    return vit_encoder
