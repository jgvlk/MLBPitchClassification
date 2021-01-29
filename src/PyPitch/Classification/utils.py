from datetime import datetime

import pandas as pd

from PyPitch.db import query_data_dictionary, query_raw_data, query_pitcher_data


class Load():

    def import_all_data(search_text):
        try:
            retd = {}

            # IMPORT DATA DICTIONARY #
            ret1, df1 = import_data_dictionary()

            retd['ret1'] = ret1

            if ret1 == 0:
                print(df1.shape)
                print(df1.head())

            # IMPORT ALL RAW DATA #
            ret2, df2 = import_raw_data()

            retd['ret2'] = ret2

            if ret2 == 0:
                print(df2.shape)
                print(df2.head())

            # IMPORT PITCHER DATA #
            ret3, df3 = import_pitcher_data(search_text)

            retd['ret3'] = ret3

            if ret3 == 0:
                print(df3.shape)
                print(df3.head())

            s = retd.values()

            if 1 in s:
                raise 'ERROR IMPORTING MODEL DATA'

            response = {'df_data_dictionary': df1, 'df_data': df2, 'df_data_pitcher': df3}

            return response

        except Exception as e:
            print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

            return e


def import_data_dictionary():
    ret, df = query_data_dictionary()

    return 0, df


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

