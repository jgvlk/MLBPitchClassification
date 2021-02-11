from datetime import datetime
import os
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

if os.name == 'posix':
    db_conn_str = r'DRIVER={/usr/local/Cellar/msodbcsql17/17.6.1.1/lib/libmsodbcsql.17.dylib};SERVER=10.0.1.3;DATABASE=MLBPitchClassification;UID=svc_MLBPitchClassification;PWD=datascience;'
else:
    db_conn_str = r'DRIVER={SQL Server};SERVER=.;DATABASE=MLBPitchClassification;TRUSTED_CONNECTION=Yes;'

db_conn_str = quote_plus(db_conn_str)
engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % db_conn_str)
Session = sessionmaker(bind=engine)


class SessionManager(object):
    def __init__(self):
        self.session = Session()
        self.engine = engine


def query_data_dictionary():
    db = SessionManager()
    conn = db.session.connection()

    sql = 'SELECT * FROM [dbo].[DataDictionary] WHERE [Include] = 1 ORDER BY [ColumnName]'

    df = pd.read_sql(sql, conn)

    db.session.commit()

    return 0, df


def query_raw_data():
    '''
    Query all raw data
    '''

    try:
        db = SessionManager()
        conn = db.session.connection()

        print('||MSG', datetime.now(), '|| QUERYING PITCHES FROM DB')

        sql = 'SELECT * FROM [dbo].[MLBPitch_2019]'

        df = pd.read_sql(sql, conn)

        db.session.commit()

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)
        print('||ERR', datetime.now(), '|| ERROR QUERYING PITCHES FROM DB')

        df = pd.DataFrame(None)

        return 1, df


def query_pitcher_data(search_text):
    '''
    Query pitch data for a specific pitcher
    '''

    try:
        db = SessionManager()
        conn = db.session.connection()

        sql = 'SELECT * FROM [dbo].[MLBPitch_2019] WHERE [PitcherFullName] LIKE ?'

        player_name = search_text
        query_arg = '%' + player_name + '%'

        print('||MSG', datetime.now(), '|| QUERYING PITCHER RECORD FOR:', player_name)

        df = pd.read_sql(sql, conn, params=[query_arg])

        return 0, df

    except Exception as e:

        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)
        print('||ERR', datetime.now(), '|| ERROR QUERYING PITCHER DATA FROM DB')

        df = pd.DataFrame(None)

        return 1, df

