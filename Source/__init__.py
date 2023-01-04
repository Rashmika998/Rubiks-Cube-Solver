import os
import sys

# Read the install dir to know where training data is
from .config import install_dir
sys.path.append(install_dir)

from .Cube import plt
from .Cube import Cube
from .Solver import Solver
from .SolutionGallery import SolutionGallery
#from .ManualSquareExtractor import ManualSquareExtractor
from .end_to_end import main
