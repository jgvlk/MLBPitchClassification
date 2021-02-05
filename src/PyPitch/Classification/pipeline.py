from pathlib import Path

from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler

from PyPitch.classification.utils import Load, FeatureEng


class Pipeline():

    def run(repo_dir='/Users/jonathanvlk/dev/MLBPitchClassification', model_version='v1'):
        repo = Path(repo_dir)
        version = model_version
        
        # TO DO: Parameterize, make optional to import one player
        data = Load.import_all_data('Lester')

        df_data_dictionary = data['df_data_dictionary']
        df_data = data['df_data']

        df_data_R = df_data.loc[df_data['PitcherThrows'] == 'R']

        # FEATURES #
        l_cols = []
        for i in df_data.columns:
            l_cols.append(i)

        l_keep_cols = ['ID', 'PitcherThrows']
        for i,r in df_data_dictionary['ColumnName'].iteritems():
            l_keep_cols.append(r)

        l_drop_cols = list(set(l_cols) - set(l_keep_cols))
        l_drop_cols.append('y0')
        # l_drop_cols.append('pfx_x')
        # l_drop_cols.append('pfx_z')
        l_drop_cols.append('Break_y')
        l_drop_cols.append('BreakAngle')
        l_drop_cols.append('BreakLength')
        # l_drop_cols.append('StartSpeed')
        # l_drop_cols.append('EndSpeed')
        l_drop_cols.append('sz_bot')
        l_drop_cols.append('sz_top')
        l_drop_cols.append('PitcherThrows')

        df = df_data_R.drop(labels=l_drop_cols, axis=1)

        features = []
        cols = df.columns

        for i in cols:
            if i != 'ID':
                features.append(i)

        # YEO-JOHNSON POWER TRANSFORMATION #
        yj_lam = []
        for i in features:
            df[i], l = yeojohnson(df[i])
            col_lambda = [i, l]
            yj_lam.append(col_lambda)

        # STANDARDIZE & NORMALIZE #
        df = FeatureEng.var_scaling(df, features, tag='')
        df = FeatureEng.var_scaling(df, features, transformer=MinMaxScaler(), tag='')

        ## TRAIN/TEST SPLIT #
        train_size = int(len(df) * .7)
        train, test = df.loc[1:train_size], df.loc[train_size:]

        return train, test, features

