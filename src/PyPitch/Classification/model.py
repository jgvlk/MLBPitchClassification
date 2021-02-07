from datetime import datetime

import numpy as np
import pandas as pd

from PyPitch.classification.pipeline import Pipeline
from PyPitch.classification.lib import KMeansModel
from PyPitch.db import SessionManager


# SET PARAMETERS #
repo = '/Users/jonathanvlk/dev/MLBPitchClassification'
version = 'v1'
test_ratio = .7
outlier_std_thresh = 6
pca_alpha = .01





# INSTANTIATE PIPELINES AND CLUSTERING MODELS #
pipeline = Pipeline()
pipelineR = Pipeline()
pipelineL = Pipeline()

model = KMeansModel(n_clust=7)
modelR = KMeansModel(n_clust=7)
modelL = KMeansModel(n_clust=7)





# LOAD RAW DATA AND SUMMARIZE #
# Pipeline.load_model_data(repo, version)
df = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df.pkl')
df_R = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_R.pkl')
df_L = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/df_L.pkl')
features = pd.read_pickle('/Users/jonathanvlk/dev/MLBPitchClassification/src/PyPitch/output/v1/data/features.pkl')
features = list(features[0])





# TRAIN/TEST SPLIT AND SUMMARIZE #
df_train, df_test = model.split(df, test_ratio)
df_R_train, df_R_test = modelR.split(df_R, test_ratio)
df_L_train, df_L_test = modelL.split(df_L, test_ratio)





# TRANSFORM #
df_train = pipeline.fit_transform_data(df_train, features)
df_R_train = pipelineR.fit_transform_data(df_R_train, features)
df_L_train = pipelineL.fit_transform_data(df_L_train, features)





# CHECK FOR & HANDLE OUTLIERS #
df_train = model.remove_outliers(df_train, features, outlier_std_thresh)
df_R_train = modelR.remove_outliers(df_R_train, features, outlier_std_thresh)
df_L_train = modelL.remove_outliers(df_L_train, features, outlier_std_thresh)





# PCA #
df_train = model.pca_fit(df_train, features)
df_R_train = modelR.pca_fit(df_R_train, features)
df_L_train = modelL.pca_fit(df_L_train, features)

vars = model.pca.explained_variance_ratio_
varsR = modelR.pca.explained_variance_ratio_
varsL = modelL.pca.explained_variance_ratio_

cumsum = np.cumsum(vars)
cumsumR = np.cumsum(varsR)
cumsumL = np.cumsum(varsL)

n_components = model.num_pca_components(cumsum, pca_alpha)
n_components_R = modelR.num_pca_components(cumsumR, pca_alpha)
n_components_L = modelL.num_pca_components(cumsumL, pca_alpha)

model.pca_var_coverage(cumsum)
modelR.pca_var_coverage(cumsumR)
modelL.pca_var_coverage(cumsumL)

model.elbow_plot(df_train, [1, n_components])
modelR.elbow_plot(df_R_train, [1, n_components_R])
modelL.elbow_plot(df_L_train, [1, n_components_L])





# CLUSTERING #
labels, centers = model.run_kmeans(df_train)
labelsR, centersR = modelR.run_kmeans(df_R_train)
labelsL, centersL = modelL.run_kmeans(df_L_train)

df_labels = pd.DataFrame(labels, columns=['Label'])
df_labelsR = pd.DataFrame(labelsR, columns=['Label'])
df_labelsL = pd.DataFrame(labelsL, columns=['Label'])

df_train = pd.DataFrame(df_train, columns=features)
df_R_train = pd.DataFrame(df_R_train, columns=features)
df_L_train = pd.DataFrame(df_L_train, columns=features)

df_train_labeled = df_train.join(df_labels)
df_R_train_labeled = df_R_train.join(df_labelsR)
df_L_train_labeled = df_L_train.join(df_labelsL)





# LABEL TEST DATA #






# OUTPUT RESULTS TO MSSQL #
db = SessionManager()
conn = db.session.connection()

table_name_train = 'Model' + version + '_Train_All'
table_name_train_R = 'Model' + version + '_Train_R'
table_name_train_L = 'Model' + version + '_Train_L'

df_train_labeled.to_sql()

