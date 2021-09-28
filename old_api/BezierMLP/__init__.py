from .loader import MLPDatum, inp_x, MatLoader as MLPMatLoader
from .model import BezierMLPModel
from .loader import MLPLoaderNode as DatumNode
from .solver import Solver
from .loader import  MLPLoaderNode as DN

Solver.train = Solver.train_extended
