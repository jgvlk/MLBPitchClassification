from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_profiling
import seaborn as sns

from PyPitch.db import query_data_dictionary, query_raw_data, query_pitcher_data


class Load():

    def import_all_data(search_text):
        try:
            retd = {}

            # IMPORT DATA DICTIONARY #
            ret1, df1 = import_data_dictionary()

            retd['ret1'] = ret1

            if ret1 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df1.shape)
                print('||MSG', datetime.now(), '|| HEAD:', df1.head())

            # IMPORT ALL RAW DATA #
            ret2, df2 = import_raw_data()

            retd['ret2'] = ret2

            if ret2 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df2.shape)
                print('||MSG', datetime.now(), '|| HEAD:', df2.head())

            # IMPORT PITCHER DATA #
            ret3, df3 = import_pitcher_data(search_text)

            retd['ret3'] = ret3

            if ret3 == 0:
                print('||MSG', datetime.now(), '|| SHAPE:', df3.shape)
                print('||MSG', datetime.now(), '|| HEAD:', df3.head())

            s = retd.values()

            if 1 in s:
                raise 'ERROR IMPORTING MODEL DATA'

            response = {'df_data_dictionary': df1, 'df_data': df2, 'df_data_pitcher': df3}

            return response

        except Exception as e:
            print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

            return e


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

