from .loader import AutoencoderCNNDatum
from .model import AutoencoderCNNModel
from .loader import AutoencoderCNNShallowDatum

load_data_from = AutoencoderCNNShallowDatum.load_data_from

del AutoencoderCNNShallowDatum

