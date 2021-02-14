from datetime import datetime
from pathlib import Path
from pprint import pprint

import pandas as pd

from PyPitch.classification.lib import EDA


# SET PARAMETERS #
repo = Path('/Users/jonathanvlk/dev/MLBPitchClassification')
version = 'v1'
output_dir_data = repo / 'src/PyPitch/output' / version / 'data'
output_dir_viz = repo / 'src/PyPitch/output' / version / 'viz'





# LOAD RAW DATA AND SUMMARIZE #
df = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df.pkl')
df_R = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_R.pkl')
df_L = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_L.pkl')
features = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/features.pkl')
features = list(features[0])





# CHECK NULLS #
null_cols = EDA.null_check(df)
null_cols_R = EDA.null_check(df_R)
null_cols_L = EDA.null_check(df_L)

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
out_file_describe = output_dir_data / 'PandasDescribe.csv'
EDA.describe(df, out_file_describe)

out_file_describe = output_dir_data / 'PandasDescribe_R.csv'
EDA.describe(df_R, out_file_describe)

out_file_describe = output_dir_data / 'PandasDescribe_L.csv'
EDA.describe(df_L, out_file_describe)





# PANDAS PROFILING #
out_file_profile = output_dir_data / 'PandasProfile.html'
EDA.profile(df, out_file_profile)

out_file_profile = output_dir_data / 'PandasProfile_R.html'
EDA.profile(df_R, out_file_profile)

out_file_profile = output_dir_data / 'PandasProfile_L.html'
EDA.profile(df_L, out_file_profile)





# FEATURE DENSITY PLOTS #
EDA.feature_density_plot(df)
EDA.feature_density_plot(df_R)
EDA.feature_density_plot(df_L)





# CORRELATION ANALYSIS #
EDA.correlation_analysis(df, features)
EDA.correlation_analysis(df_R, features)
EDA.correlation_analysis(df_L, features)

top10_neg, top10_pos = EDA.correlation_rank(df, features)
top10_neg_R, top10_pos_R = EDA.correlation_rank(df_R, features)
top10_neg_L, top10_pos_L = EDA.correlation_rank(df_L, features)

d_corr = {}
for i in range(10):
    d_corr['neg_{}'.format(i)] = {'all data': top10_neg[i], 'righties': top10_neg_R[i], 'lefties': top10_neg_L[i]}
    d_corr['pos_{}'.format(i)] = {'all data': top10_pos[i], 'righties': top10_pos_R[i], 'lefties': top10_pos_L[i]}

pprint(d_corr)

