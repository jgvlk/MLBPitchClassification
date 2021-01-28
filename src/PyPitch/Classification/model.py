from datetime import datetime

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pprint import pprint
import statsapi

from PyPitch.classification.utils import load
from PyPitch.classification.viz import show_release_point, show_pitch_location


### [1] EDA & TRANSFORMATIONS ###
#################################

data = load()

