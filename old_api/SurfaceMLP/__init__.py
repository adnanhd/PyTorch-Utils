from .loader import SurfaceMLPDatum, SurfaceMLPShallowDatum
from .model  import SurfaceMLPModel
load_data_from = SurfaceMLPShallowDatum.load_data_from

del SurfaceMLPShallowDatum
