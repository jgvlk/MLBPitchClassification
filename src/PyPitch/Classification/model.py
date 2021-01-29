from datetime import datetime

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pprint import pprint
import statsapi

from PyPitch.classification.utils import Load
from PyPitch.classification.viz import show_release_point, show_pitch_location


### [1] LOAD DATA ###
#####################

data = Load.import_all_data('Lester')

df_data_dictionary = data['df_data_dictionary']
df_data = data['df_data']
df_data_pitcher = data['df_data_pitcher']





### [2] EDA & TRANSFORMATIONS ###
#################################


l_cols = []
for i in df_data.columns:
    l_cols.append(i)

l_keep_cols = ['ID']
for i,r in df_data_dictionary['ColumnName'].iteritems():
    l_keep_cols.append(r)

l_drop_cols = list(set(l_cols) - set(l_keep_cols))

df_data_model = df_data.drop(labels=l_drop_cols, axis=1)
features = df_data_model.columns


df_data

