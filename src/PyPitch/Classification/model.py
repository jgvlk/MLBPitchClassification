from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
import pandas as pd
# import pandas_profiling
# from pprint import pprint
import seaborn as sns
# import statsapi

from PyPitch.classification.utils import Load, EDA
# from PyPitch.classification.viz import show_release_point, show_pitch_location


### SET VARS ###
################

REPO = '/Users/jonathanvlk/dev/MLBPitchClassification'
MODEL_VERSION = 'v1'

REPO = Path(REPO)
MODEL_VERSION = Path(MODEL_VERSION)

output_dir_data = REPO / 'src/PyPitch/output' / MODEL_VERSION / 'data'
output_dir_viz = REPO / 'src/PyPitch/output' / MODEL_VERSION / 'viz'





### [1] LOAD DATA ###
#####################

data = Load.import_all_data('Lester')

df_data_dictionary = data['df_data_dictionary']
df_data = data['df_data']
df_data_pitcher = data['df_data_pitcher']

df_data_R = df_data.loc[df_data['PitcherThrows'] == 'R']
df_data_L = df_data.loc[df_data['PitcherThrows'] == 'L']





### [2] EDA & TRANSFORMATIONS ###
#################################

# MODEL FEATURES #
l_cols = []
for i in df_data.columns:
    l_cols.append(i)

l_keep_cols = ['ID', 'PitcherThrows']
for i,r in df_data_dictionary['ColumnName'].iteritems():
    l_keep_cols.append(r)

l_drop_cols = list(set(l_cols) - set(l_keep_cols))
l_drop_cols.append('y0')
l_drop_cols.append('pfx_x')
l_drop_cols.append('pfx_z')
l_drop_cols.append('Break_y')
l_drop_cols.append('BreakAngle')
l_drop_cols.append('BreakLength')
l_drop_cols.append('StartSpeed')
l_drop_cols.append('EndSpeed')
l_drop_cols.append('sz_bot')
l_drop_cols.append('sz_top')
l_drop_cols.append('PitcherThrows')

df_data_model = df_data.drop(labels=l_drop_cols, axis=1)
df_data_model_R = df_data_R.drop(labels=l_drop_cols, axis=1)
df_data_model_L = df_data_L.drop(labels=l_drop_cols, axis=1)

features = []
cols = df_data_model.columns

for i in cols:
    if i != 'ID':
        features.append(i)


# CREATE SEPARATE DATASETS FOR R/L #
df_data_model_R = df_data_model.loc[df_data_model['PitcherThrows'] == 'R']
df_data_model_L = df_data_model.loc[df_data_model['PitcherThrows'] == 'L']

df_data_model.drop(labels=['PitcherThrows'], axis=1)
df_data_model_R.drop(labels=['PitcherThrows'], axis=1)
df_data_model_L.drop(labels=['PitcherThrows'], axis=1)


# CHECK NULLS #
null_cols = EDA.null_check(df_data_model)
null_cols_R = EDA.null_check(df_data_model_R)
null_cols_L = EDA.null_check(df_data_model_L)

if null_cols:
    print('||WARN', datetime.now(), '|| NULL FEATURE VALUES EXIST')
else:
    print('||MSG', datetime.now(), '|| NO NULL FEATURE VALUES EXIST')

if null_cols_R:
    print('||WARN', datetime.now(), '|| NULL FEATURE VALUES EXIST')
else:
    print('||MSG', datetime.now(), '|| NO NULL FEATURE VALUES EXIST')

if null_cols_L:
    print('||WARN', datetime.now(), '|| NULL FEATURE VALUES EXIST')
else:
    print('||MSG', datetime.now(), '|| NO NULL FEATURE VALUES EXIST')



# PANDAS DESCRIBE #
out_file_describe = output_dir_data / 'Model_RawData_Describe2.csv'
EDA.describe(df_data_model, out_file_describe)

out_file_describe = output_dir_data / 'Model_RawData_Describe2_R.csv'
EDA.describe(df_data_model_R, out_file_describe)

out_file_describe = output_dir_data / 'Model_RawData_Describe2_L.csv'
EDA.describe(df_data_model_L, out_file_describe)


# PANDAS PROFILING #
out_file_profile = output_dir_data / 'profile2.html'
EDA.profile(df_data_model, out_file_profile)

out_file_profile = output_dir_data / 'profile2_R.html'
EDA.profile(df_data_model_R, out_file_profile)

out_file_profile = output_dir_data / 'profile2_L.html'
EDA.profile(df_data_model_L, out_file_profile)


# FEATURE DENSITY PLOTS #
EDA.feature_density_plot(df_data_model)
EDA.feature_density_plot(df_data_model_R)
EDA.feature_density_plot(df_data_model_L)


# CORRELATION ANALYSIS #
EDA.correlation_analysis(df_data_model, features)
EDA.correlation_analysis(df_data_model_R, features)
EDA.correlation_analysis(df_data_model_L, features)

