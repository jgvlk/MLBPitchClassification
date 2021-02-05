from datetime import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling
from scipy.stats import zscore
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from PyPitch.db import query_data_dictionary, query_raw_data, query_pitcher_data


class EDA():

    def correlation_analysis(data, num_cols, univariate = False):
        corr = data[num_cols].corr()

        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=(11, 9))

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)

        plt.show()

    def correlation_rank(data, num_cols):
        corr = data[num_cols].corr()

        d = {}
        for var in corr:
            var_name = corr[var].name
            ix = corr[var].index
            corr_value = corr[var]

            for i, element in enumerate(ix):
                l = [var_name, element]
                l.sort()

                if var_name != element:
                    var_pair = l[0] + '__' + l[1]
                    d[var_pair] = corr_value[i]

        # Sort var pairs by correlation values
        d_sort_values = sorted(d.items(), key=lambda x: x[1], reverse=False)
        d_sort_values_rev = sorted(d.items(), key=lambda x: x[1], reverse=True)

        # Define highest positive & negative correlations
        top10_neg = list(d_sort_values)[:10]
        top10_pos = list(d_sort_values_rev)[:10]

        return top10_neg, top10_pos


    def describe(df, out_file):
        df_describe = df.describe()
        df_describe.to_csv(out_file)

    def feature_density_plot(df):
        df.plot(kind='density', subplots=True, layout=(5,5), sharex=False, figsize=(15,10))
        plt.show()

    def null_check(df):
        d_null_cols = {}
        null_check = df.isnull().sum()

        for i in range(len( null_check)):
            null_ct = None
            null_key = None

            if null_check[i] != 0:
                null_ct = null_check[i]
                null_key = null_check.keys()[i]

                d_null_cols[null_key] = null_ct

                print('||WARN', datetime.now(), '||', null_ct, 'NULL VALUES EXIST FOR', null_key)

        return d_null_cols

    def profile(df, out_file):
        profile = pandas_profiling.ProfileReport(df)
        profile.to_file(out_file)


class FeatureEng():

    def detect_outliers(col_names, data, std_thresh = 6):

        d_outliers = {}

        for column in col_names:

            # Create z_score proxy for each column
            data['z_score'] = np.absolute(zscore(data[column]))

            # Check if there are NaNs in z_score
            if data['z_score'].isnull().sum() > 0:
                print('WARNING: NaNs found in data column: {}. More analysis may be necessary to ensure there are not outliers'.format(column))

            # Determine if there are outliers, as defined by z_score threshold
            outliers = data.loc[data.z_score > std_thresh, [column, 'z_score']]

            # If there are no outliers
            if outliers.shape[0] == 0:
                print('No outliers for column {} at threshold of {} stdevs'.format(column, std_thresh))

                d_outliers[column] = []

            # If there are outliers
            else:
                print('{} outlier(s) found for column {} at threshold of {} stdevs'.format(outliers.shape[0], column, std_thresh))

                l_column_outliers = []
                for i,r in outliers.iterrows():
                    l_column_outliers.append({'index': i, 'value': r[column], 'z_score': r['z_score']})

                d_outliers[column] = l_column_outliers

            # Drop z_score from data
            data.drop('z_score', axis = 1, inplace = True)

        return d_outliers

    def var_scaling(data, column_names, transformer = StandardScaler(), tag = 'SS_'):
        scaled_df = transformer.fit_transform(data.loc[:,column_names])

        return pd.DataFrame(scaled_df, columns = [tag + col for col in column_names])


class Load():

    def import_all_data(search_text):
        try:
            retd = {}

            # IMPORT DATA DICTIONARY #
            ret1, df1 = import_data_dictionary()

            retd['ret1'] = ret1

            if ret1 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df1.shape)

            # IMPORT ALL RAW DATA #
            ret2, df2 = import_raw_data()

            retd['ret2'] = ret2

            if ret2 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df2.shape)

            # IMPORT PITCHER DATA #
            ret3, df3 = import_pitcher_data(search_text)

            retd['ret3'] = ret3

            if ret3 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df3.shape)

            s = retd.values()

            if 1 in s:
                raise 'ERROR IMPORTING MODEL DATA'

            response = {'df_data_dictionary': df1, 'df_data': df2, 'df_data_pitcher': df3}

            return response

        except Exception as e:
            print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

            return e


class Model():

    def num_pca_components(cum_sum, alpha):
        threshold = 1 - alpha
        n = 1

        for i in cum_sum:
            if i >= threshold:
                
                return n
            
            else:
                n += 1

    def pca_var_coverage(explained_var_cumsum):
        plt.figure(figsize = (20,10))
        x = np.arange(1, len(explained_var_cumsum) + 1, 1)
        plt.plot(x, explained_var_cumsum)
        
        plt.title("Variance Coverage by Principal Components")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Variance Coverage Percentage")
        plt.show()


def import_data_dictionary():
    ret, df = query_data_dictionary()

    return 0, df


def import_pitcher_data(search_text):
    '''
    Import raw pitcher data from sql instance
    '''

    try:
        print('||MSG', datetime.now(), '|| IMPORTING RAW DATASET')

        ret, df = query_pitcher_data(search_text)

        if ret == 1:
            raise Exception()

        print('||MSG', datetime.now(), '|| IMPORTED RAW DATASET SUCCESSFULLY WITH SHAPE:', df.shape)

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

        df = pd.DataFrame(None)

        return 1, df


def import_raw_data():
    '''
    Import raw dataset from sql instance
    '''

    try:
        print('||MSG', datetime.now(), '|| IMPORTING RAW DATASET')

        ret, df = query_raw_data()

        if ret == 1:
            raise Exception()

        print('||MSG', datetime.now(), '|| IMPORTED RAW DATASET SUCCESSFULLY WITH SHAPE:', df.shape)

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

        df = pd.DataFrame(None)

        return 1, df

