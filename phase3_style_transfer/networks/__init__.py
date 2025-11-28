# Commented out to avoid metric module dependency during inference
# from networks.model import RecognizeModel, WriterIdentifyModel, GlobalLocalAdversarialModel

# Import improved components
from networks.improved_layers import (
    AdaIN, ModulatedConv2d, ModulatedGBlock,
    MultiHeadCrossAttention, StyleContentCrossAttention,
    TextTransformerEncoder, BiGRUEncoder,
    SinusoidalPositionalEncoding, LearnablePositionalEncoding,
    ContrastiveStyleLoss, MultiScaleStyleFusion,
    ImprovedSelfAttention
)

from networks.BigGAN_networks import Generator, ImprovedGenerator, Discriminator
from networks.multi_scale_discriminator import MultiScaleDiscriminator, ProgressiveDiscriminator

all_models = {
    'gl_adversarial_model': GlobalLocalAdversarialModel,
    'recognize_model': RecognizeModel,
    'identifier_model': WriterIdentifyModel
}


# def get_model(name):
#     return all_models[name]