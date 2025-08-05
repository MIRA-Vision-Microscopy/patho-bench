from typing import Dict, Mapping, Optional, Tuple, Union, Type, cast
import inspect
import warnings

import torch.nn as nn
from torch import Tensor
import torch
from trident.slide_encoder_models.load import CustomSlideEncoder, BaseSlideEncoder
from .agata_slide_encoder import AgataModel, FCHeadConfig

# registry dict, initially empty
_CUSTOM_ENCODERS: Dict[str, Type[nn.Module]] = {}

# decorator factory to register encoders
def register_encoder(name: str):
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        _CUSTOM_ENCODERS[name] = cls
        return cls
    return decorator

def custom_encoder_factory(
    model_name: str,
    *,
    pretrained: bool = True,
    freeze: bool = False,
    **model_kwargs
) -> nn.Module:
    try:
        EncoderCls = _CUSTOM_ENCODERS[model_name]
    except KeyError:
        raise ValueError(f"Unknown custom encoder '{model_name}'")

    # Find what __init__ actually accepts:
    sig = inspect.signature(EncoderCls.__init__)
    # drop 'self' and any unsupported kwargs
    init_kwargs = {
        k: v
        for k, v in {
            **model_kwargs,
            "pretrained": pretrained,
            "freeze": freeze,
        }.items()
        if k in sig.parameters and k != "self"
    }

    # Warn if we silently dropped flags
    if ("pretrained" in init_kwargs) is False and pretrained:
        warnings.warn(f"Ignoring pretrained=True for {model_name}", UserWarning)
    if ("freeze" in init_kwargs) is False and freeze:
        warnings.warn(f"Ignoring freeze=True for {model_name}", UserWarning)

    print(init_kwargs)

    return EncoderCls(**init_kwargs)

# by decorating the class registers itself
@register_encoder("agata")
class AgataSlideEncoder(BaseSlideEncoder):
    def __init__(self,
            in_features: int,
            layer1_out_features: int,
            layer2_out_features: int,
            activation: nn.Module = torch.nn.GELU(),
            label_name_fclayer_head_config: Mapping[str, FCHeadConfig] = None,
            scaled_attention: bool = False,
            absolute_attention: bool = False,
            n_attention_queries: int = 1,
            padding_indicator: int = 1,
            freeze: bool = False):
        """
        Agata initialization.
        """
        super().__init__(
            in_features = in_features,
            layer1_out_features = layer1_out_features,
            layer2_out_features = layer2_out_features,
            activation = activation,
            scaled_attention = scaled_attention,
            absolute_attention = absolute_attention,
            n_attention_queries = n_attention_queries,
            padding_indicator = padding_indicator,
            freeze = freeze)

    def _build(
            self,
            in_features: int,
            layer1_out_features: int,
            layer2_out_features: int,
            activation: nn.Module,
            scaled_attention: bool = False,
            absolute_attention: bool = False,
            n_attention_queries: int = 1,
            padding_indicator: int = 1,
        ) -> Tuple[torch.nn.ModuleDict, torch.dtype, int]:
        

        self.enc_name = 'agata'

        model = AgataModel(
            in_features = in_features,
            layer1_out_features = layer1_out_features,
            layer2_out_features = layer2_out_features,
            activation = activation,
            label_name_fclayer_head_config = None,
            scaled_attention = scaled_attention,
            absolute_attention = absolute_attention,
            n_attention_queries = n_attention_queries,
            padding_indicator = padding_indicator
        )

        precision = torch.float32
        embedding_dim = layer2_out_features
        return model, precision, embedding_dim

    def forward(self, batch, device = 'cuda'):
        x = batch['features'].to(device)
        # x = (batch_size, sequence_dim, feature_dim)
        z,_ = self.model.forward(x, padding_masks = None)
        return z
