from datetime import datetime

from PyPitch.classification.pipeline import Pipeline
from PyPitch.classification.utils import FeatureEng


df_train, df_test, features = Pipeline.run()


# CHECK FOR & HANDLE OUTLIERS #
outliers = FeatureEng.detect_outliers(features, df_train, 5)

delete = []
for i in outliers:
    if outliers[i]:
        for i in outliers[i]:
            delete.append(i['index'])


df_train = df_train.drop(df_train.index[delete])

