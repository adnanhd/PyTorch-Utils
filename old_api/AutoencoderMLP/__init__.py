from .loader import AutoencoderMLPDatum, AutoencoderMLPShallowDatum
from .model  import AutoencoderMLPModel
load_data_from = AutoencoderMLPShallowDatum.load_data_from

del AutoencoderMLPShallowDatum
del loader
del model

__all__ = [AutoencoderMLPDatum, AutoencoderMLPModel, load_data_from]
