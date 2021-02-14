import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import event

from PyPitch.db import SessionManager
from PyPitch.classification.lib import Result


db = SessionManager()
conn = db.session.connection()
_engine = db.engine

@event.listens_for(_engine, 'before_cursor_execute')
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    if executemany:
        cursor.fast_executemany = True


sql_v1 = '''
SELECT
    test.[Label_R]
    ,p.[pfx_x]
    ,p.[pfx_z]
FROM
    [output].[Model_v1_Test_R_Labeled] test
    JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
'''

sql_v2 = '''
SELECT
    test.[Label_R]
    ,p.[pfx_x]
    ,p.[pfx_z]
FROM
    [output].[Model_v2_Test_R_Labeled] test
    JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
'''

sql_v3 = '''
SELECT
    test.[Label_R]
    ,p.[pfx_x]
    ,p.[pfx_z]
FROM
    [output].[Model_v3_Test_R_Labeled] test
    JOIN [dbo].[MLBPitch_2019] p ON test.[ID] = p.[ID]
'''

df_pfx_v1 = pd.read_sql(sql_v1, conn)
df_pfx_v2 = pd.read_sql(sql_v2, conn)
df_pfx_v3 = pd.read_sql(sql_v3, conn)


Result.pfx_scatter_alllabels(df_pfx_v1, 'Label_R')
Result.pfx_scatter_alllabels(df_pfx_v2, 'Label_R')
Result.pfx_scatter_alllabels(df_pfx_v3, 'Label_R')

Result.pfx_scatter_bylabel(df_pfx_v1, 'Label_R')
Result.pfx_scatter_bylabel(df_pfx_v2, 'Label_R')
Result.pfx_scatter_bylabel(df_pfx_v3, 'Label_R')

