import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from PyPitch.classification.pipeline import Pipeline
from PyPitch.classification.utils import FeatureEng, EDA, Model


# df_train, df_test, features = Pipeline.run()

df_train = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/df_train.pkl')
df_test = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/df_test.pkl')
features = ['ax0', 'ay0', 'az0', 'StartSpeed', 'EndSpeed', 'pfx_x', 'pfx_z', 'px', 'pz', 'vx0', 'vy0', 'vz0', 'x0', 'z0']

# CHECK FOR & HANDLE OUTLIERS #
outliers = FeatureEng.detect_outliers(features, df_train, 5)

delete = []
for i in outliers:
    if outliers[i]:
        for i in outliers[i]:
            delete.append(i['index'])


df_train = df_train.drop(df_train.index[delete])


# PCA #
EDA.correlation_analysis(df_train, features)

pca = PCA()
pca_fit = pca.fit_transform(df_train)

vars = pca.explained_variance_ratio_

cum_sum = np.cumsum(vars)

n_components = Model.num_pca_components(cum_sum, .01)

