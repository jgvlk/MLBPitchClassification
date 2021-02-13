import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from sqlalchemy import event

from PyPitch.db import Session, SessionManager


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


def pfx_scatter_labels(data, label_col_name):
    labels = list(data[label_col_name].drop_duplicates())
    colors = ['red', 'green', 'blue', 'purple']

    fig, a = plt.subplots()
    for color, label in zip(colors, labels):
        pfx_x = np.array(data['pfx_x'].loc[data['Label_R']==label])
        pfx_z = np.array(data['pfx_z'].loc[data['Label_R']==label])
        a.scatter(pfx_x, pfx_z, c=color, label=label)

    a.legend()
    a.grid(True)

    plt.show()


pfx_scatter_labels(df_pfx_v1, 'Label_R')
pfx_scatter_labels(df_pfx_v2, 'Label_R')
pfx_scatter_labels(df_pfx_v3, 'Label_R')

