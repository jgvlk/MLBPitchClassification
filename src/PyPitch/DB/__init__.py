from datetime import datetime
from urllib.parse import quote_plus

import pandas as pd
import statsapi
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


db_conn_str = r'DRIVER={/usr/local/Cellar/msodbcsql17/17.6.1.1/lib/libmsodbcsql.17.dylib};SERVER=10.0.1.3;DATABASE=MLBPitchClassification;UID=svc_MLBPitchClassification;PWD=datascience;'
db_conn_str = quote_plus(db_conn_str)

engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % db_conn_str)

Session = sessionmaker(bind=engine)

class SessionManager(object):
    def __init__(self):
        self.session = Session()
        self.engine = engine


def query_raw_data():
    '''
    Query raw dataset
    Returns: ret, df
        - ret: return code, 0 for success, 1 for error
        - df: dataframe containing raw data, None if error

    TO DO:
    - Give option to select top n rows
    '''

    try:
        db = SessionManager()
        conn = db.session.connection()

        print('||MSG', datetime.now(), '|| QUERYING PITCHES FROM DB')

        sql = 'SELECT TOP 10000 * FROM [dbo].[MLBPitch_2019] ORDER BY NEWID()'

        df = pd.read_sql(sql, conn)

        db.session.commit()

        return 0, df

    except Exception as e:
        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)
        print('||ERR', datetime.now(), '|| ERROR QUERYING PITCHES FROM DB')

        df = pd.DataFrame(None)

        return 1, df


def query_pitcher_data(search_text, season):
    '''
    Query pitch data for a specific pitcher
    '''

    try:
        try:
            print('||MSG', datetime.now(), '|| FETCHING PITCHER''S PLAYER RECORD')

            player = statsapi.lookup_player(search_text, season=season)

            player_id = player[0]['id']
            player_name = player[0]['firstLastName']
            
        except Exception as e:
            print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)

            raise

        db = SessionManager()
        conn = db.session.connection()

        sql = 'SELECT * FROM [dbo].[MLBPitch_2019] WHERE [Pitcher] = ?'

        print('||MSG', datetime.now(), '|| QUERYING PITCHER RECORD FOR:', player_name)

        df = pd.read_sql(sql, conn, params=[player_id])

        return 0, df

    except Exception as e:

        print('||ERR', datetime.now(), '|| ERROR MESSAGE:', e)
        print('||ERR', datetime.now(), '|| ERROR QUERYING PITCHER DATA FROM DB')

        df = pd.DataFrame(None)

        return 1, df

