from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from pprint import pprint
from sqlalchemy import event
from sqlalchemy.dialects import mssql

from PyPitch.DB.ConnManager import SessionManager


def GetRawDataset():

    db = SessionManager()
    conn = db.session.connection()
    _engine = db.engine

    print('||MSG', datetime.now(), '|| QUERYING PITCHES FROM DB')

    sql = '''
        SELECT
            [PitchID]
            ,[MLBGameID]
            ,[PlayGUID]
            ,[ID]
            ,[EventNum]
            ,[Des]
            ,[DesEs]
            ,[x0]
            ,[y0]
            ,[z0]
            ,[vx0]
            ,[vy0]
            ,[vz0]
            ,[ax0]
            ,[ay0]
            ,[az0]
            ,[px]
            ,[pz]
            ,[pfx_x]
            ,[pfx_z]
            ,[Break_y]
            ,[BreakAngle]
            ,[BreakLength]
            ,[StartSpeed]
            ,[EndSpeed]
            ,[Type]
            ,[PitchType]
        FROM
            [dbo].[MLBPitch_2019]
    '''

    df = pd.read_sql(sql, conn)

    db.session.commit()

    return df

