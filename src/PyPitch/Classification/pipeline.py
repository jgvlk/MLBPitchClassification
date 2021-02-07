from pathlib import Path

import pandas as pd
from scipy.stats import yeojohnson
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler

from PyPitch.classification.lib import Load


class Pipeline():

    def __init__(self):
        '''
        Define Pipeline() instance transformers/scalers
        '''

        self.pt = PowerTransformer(method='yeo-johnson', standardize=False,)
        self.ss = StandardScaler()
        self.mm = MinMaxScaler()


    def fit_transform_data(self, data, features):
        '''
        This method applies an optimal power transformation via Yeo-Johnson method, then standardizes and normalizes
        '''

        _pt = self.pt
        _ss = self.ss
        _mm = self.mm

        # YEO-JOHNSON POWER TRANSFORMATION #
        yeojt = _pt.fit(data[features])
        yeojt = _pt.transform(data[features])
        df_yeojt = pd.DataFrame(yeojt, columns=features)

        # STANDARDIZE & NORMALIZE #
        df_std = pd.DataFrame(_ss.fit_transform(df_yeojt[features]), columns=features)
        df_std_norm = pd.DataFrame(_mm.fit_transform(df_std[features]), columns=features)

        # FORMAT RETURN DATA #
        df = data.drop(labels=features, axis=1)
        df = df.join(df_std_norm)

        return df


    def load_model_data(repo_dir, model_version):
        '''
        TO DO:
        - Check if pkl files exist, don't load if they already exist
        - Parameterize player data import, make optional to import one player
        '''

        repo = Path(repo_dir)
        data_output = repo / 'src/PyPitch/output' / model_version / 'data'

        # LOAD RAW DATA #
        data = Load.import_all_data('Lester')

        df_data_dictionary = data['df_data_dictionary']
        df_data = data['df_data']

        df_data_R = df_data.loc[df_data['PitcherThrows'] == 'R']
        df_data_R = df_data_R.reset_index()
        df_data_R = df_data_R.drop(labels=['index'], axis=1)

        df_data_L = df_data.loc[df_data['PitcherThrows'] == 'L']
        df_data_L = df_data_L.reset_index()
        df_data_L = df_data_L.drop(labels=['index'], axis=1)

        # DEFINE FEATURES #
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

        df = df_data.drop(labels=l_drop_cols, axis=1)
        df_R = df_data_R.drop(labels=l_drop_cols, axis=1)
        df_L = df_data_L.drop(labels=l_drop_cols, axis=1)

        features = []
        cols = df.columns

        for i in cols:
            if i != 'ID':
                features.append(i)

        df_features = pd.DataFrame(features)

        df_out_file = data_output / 'df.pkl'
        df_R_out_file = data_output / 'df_R.pkl'
        df_L_out_file = data_output / 'df_L.pkl'
        df_features_out_file = data_output / 'features.pkl'

        df.to_pickle(df_out_file)
        df_R.to_pickle(df_R_out_file)
        df_L.to_pickle(df_L_out_file)
        df_features.to_pickle(df_features_out_file)

        return 0


    def transform_data():
        '''
        '''


    def rev_transform_data():
        '''
        '''

