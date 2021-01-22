from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from PyPitch.classification.utils import import_raw_dataset
from PyPitch.classification.viz import show_release_point, show_pitch_location





### [1] EDA & TRANSFORMATIONS ###
#################################

# IMPORT DATASET #
ret, df = import_raw_dataset()


# VIZ: RELEASE POINT #
########
# TO DO:
#   - Look at how release points differ for an individual pitcher
#       - May need to add player lookups to db
########
x_release = df['x0']
y_release = df['y0']
z_release = df['z0']

x_location = df['px']
z_location = df['pz']

show_release_point(x_release, y_release, z_release, 'b')
show_pitch_location(x_location, z_location, 'r')

