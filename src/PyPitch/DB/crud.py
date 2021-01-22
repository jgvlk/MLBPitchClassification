from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd

from PyPitch.db.conn_manager import SessionManager


def get_raw_dataset():
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

